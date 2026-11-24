import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from lanllm.model_sources import (
    download_model,
    hf_model_files,
    ollama_manifest,
    parse_source,
)


class SourceTests(unittest.TestCase):
    def test_source_names_and_urls_choose_backend(self) -> None:
        self.assertEqual(parse_source("Qwen/Qwen2.5-14B-Instruct")["backend"], "vllm")
        self.assertEqual(
            parse_source("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")[
                "name"
            ],
            "meta-llama/Llama-3.1-8B-Instruct",
        )
        self.assertEqual(
            parse_source("https://ollama.com/library/qwen2.5:0.5b")["ref"],
            "ollama:qwen2.5:0.5b",
        )
        self.assertEqual(parse_source("ollama:qwen2.5")["ref"], "ollama:qwen2.5:latest")
        self.assertEqual(
            parse_source("hf:owner/repo", "llama.cpp", "model.Q4_K_M.gguf")["ref"],
            "hf:owner/repo@model.Q4_K_M.gguf",
        )
        self.assertEqual(
            parse_source("hf:owner/repo@model.Q4_K_M.gguf")["backend"],
            "llama.cpp",
        )

    def test_rejects_paths_and_backend_mismatch(self) -> None:
        for source, backend, filename in (
            ("ollama:../x", None, ""),
            ("hf:owner/../repo", None, ""),
            ("owner/repo", "llama.cpp", "../../file.gguf"),
            ("ollama:qwen2.5", "vllm", ""),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_source(source, backend, filename)

    def test_ollama_manifest_chooses_model_weight_layer(self) -> None:
        spec = parse_source("ollama:qwen2.5:0.5b")
        manifest = {
            "layers": [
                {"mediaType": "application/vnd.ollama.image.template", "size": 12},
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "size": 397807936,
                    "digest": "sha256:" + "a" * 64,
                },
            ]
        }
        with patch("lanllm.model_sources.urllib.request.urlopen") as open_url:
            open_url.return_value.__enter__.return_value = SimpleNamespace(
                read=lambda: json.dumps(manifest).encode()
            )
            layer = ollama_manifest(spec)
        self.assertEqual(layer["size"], 397807936)

    def test_ollama_manifest_rejects_unavailable_projector(self) -> None:
        spec = parse_source("ollama:qwen2.5:0.5b")
        manifest = {
            "layers": [
                {"mediaType": "application/vnd.ollama.image.model"},
                {"mediaType": "application/vnd.ollama.image.projector"},
            ]
        }
        with patch("lanllm.model_sources.urllib.request.urlopen") as open_url:
            open_url.return_value.__enter__.return_value = SimpleNamespace(
                read=lambda: json.dumps(manifest).encode()
            )
            with self.assertRaisesRegex(ValueError, "extra weight files"):
                ollama_manifest(spec)

    def test_download_rejects_oversize_before_writing(self) -> None:
        spec = parse_source("ollama:qwen2.5:0.5b")
        with TemporaryDirectory() as directory:
            with patch(
                "lanllm.model_sources.ollama_manifest", return_value={"size": 500}
            ):
                with self.assertRaisesRegex(ValueError, "online GPUs allow"):
                    download_model(spec, Path(directory), None, 400, lambda *_: None)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_hf_gguf_prefers_single_q4_k_m_file(self) -> None:
        spec = parse_source("hf:owner/repo", "llama.cpp")
        siblings = [
            SimpleNamespace(rfilename="model.Q8_0.gguf", size=800),
            SimpleNamespace(rfilename="model.Q4_K_M.gguf", size=400),
        ]
        hub = SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(siblings=siblings)
        )
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            files, size = hf_model_files(spec, None)
        self.assertEqual(files, ["model.Q4_K_M.gguf"])
        self.assertEqual(size, 400)
        self.assertEqual(spec["ref"], "hf:owner/repo@model.Q4_K_M.gguf")

    def test_hf_gguf_only_repository_selects_llama_automatically(self) -> None:
        spec = parse_source("owner/repo")
        siblings = [SimpleNamespace(rfilename="model.Q4_K_M.gguf", size=400)]
        hub = SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(siblings=siblings)
        )
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            _, size = hf_model_files(spec, None)
        self.assertEqual(size, 400)
        self.assertEqual(spec["backend"], "llama.cpp")

    def test_hf_checkpoint_skips_gguf_alternatives(self) -> None:
        spec = parse_source("owner/repo")
        siblings = [
            SimpleNamespace(rfilename="model.safetensors", size=400),
            SimpleNamespace(rfilename="model.Q4_K_M.gguf", size=200),
        ]
        ignored = []

        def snapshot_download(**kwargs):
            ignored.extend(kwargs["ignore_patterns"])
            Path(kwargs["local_dir"], "config.json").write_text("{}")
            Path(kwargs["local_dir"], "model.safetensors").write_bytes(b"x")

        hub = SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(siblings=siblings),
            snapshot_download=snapshot_download,
        )
        with TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                download_model(spec, Path(directory), None, 1_000, lambda *_: None)
        self.assertIn("*.gguf", ignored)


if __name__ == "__main__":
    unittest.main()
