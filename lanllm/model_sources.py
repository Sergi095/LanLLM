"""Download Hugging Face checkpoints and GGUF models into the leader's store."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
OLLAMA_NAME = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?"
    r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?\Z"
)
GGUF_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.gguf\Z", re.I)
REGISTRY = "https://registry.ollama.ai/v2"


def parse_source(source: str, backend: str | None = None, filename: str = "") -> dict:
    """Return a canonical source descriptor. Plain owner/repo means HF."""
    raw = source.strip()
    if raw.startswith("https://"):
        url = urlparse(raw)
        if url.query or url.fragment or url.params:
            raise ValueError("Model URL must not contain a query or fragment")
        if url.netloc == "huggingface.co":
            raw = "hf:" + url.path.strip("/")
        elif url.netloc == "ollama.com" and url.path.startswith("/library/"):
            raw = "ollama:" + url.path.removeprefix("/library/").strip("/")
        else:
            raise ValueError("Use a Hugging Face repository or Ollama library URL")
    if raw.startswith("ollama:"):
        name = raw.removeprefix("ollama:")
        if not OLLAMA_NAME.fullmatch(name) or ".." in name:
            raise ValueError("Invalid Ollama model name")
        if backend not in (None, "llama.cpp") or filename:
            raise ValueError("Ollama library models use llama.cpp and no --file")
        if ":" not in name:
            name += ":latest"
        repository, tag = name.rsplit(":", 1)
        registry_repo = repository if "/" in repository else f"library/{repository}"
        return {
            "source": "ollama",
            "name": name,
            "backend": "llama.cpp",
            "ref": f"ollama:{name}",
            "repository": registry_repo,
            "tag": tag,
            "relative_path": f"gguf/ollama/{registry_repo}/{tag}/model.gguf",
        }
    if raw.startswith("hf:") and "@" in raw:
        raw, specified_file = raw.rsplit("@", 1)
        if filename and filename != specified_file:
            raise ValueError("Conflicting GGUF filenames")
        filename = specified_file
        if backend == "vllm":
            raise ValueError("GGUF references use llama.cpp")
        backend = "llama.cpp"
    name = raw.removeprefix("hf:")
    if not HF_ID.fullmatch(name) or ".." in name:
        raise ValueError("Use owner/repository, a Hugging Face URL, or ollama:name:tag")
    auto_backend = backend is None and not filename
    backend = backend or ("llama.cpp" if filename else "vllm")
    if backend not in {"vllm", "llama.cpp"}:
        raise ValueError("Backend must be vllm or llama.cpp")
    if backend == "vllm":
        if filename:
            raise ValueError("--file is only for llama.cpp GGUF models")
        return {
            "source": "hf",
            "name": name,
            "backend": backend,
            "auto_backend": auto_backend,
            "ref": name,
            "relative_path": f"models/{name}",
        }
    if filename and (not GGUF_FILE.fullmatch(filename) or ".." in filename):
        raise ValueError("--file must be a GGUF filename without directories")
    return {
        "source": "hf",
        "name": name,
        "backend": backend,
        "auto_backend": auto_backend,
        "filename": filename,
        "ref": f"hf:{name}@{filename}" if filename else f"hf:{name}",
        "relative_path": f"gguf/hf/{name}/{filename}" if filename else "",
    }


def ollama_manifest(spec: dict) -> dict:
    def fetch(identifier: str) -> dict:
        url = f"{REGISTRY}/{spec['repository']}/manifests/{identifier}"
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)

    manifest = fetch(spec["tag"])
    if manifest.get("manifests"):
        choices = [
            item
            for item in manifest["manifests"]
            if item.get("format") == "gguf"
            or item.get("runner") in {"ggml", "llamacpp"}
        ]
        if len(choices) != 1:
            raise ValueError("Ollama model has no unambiguous GGUF variant")
        manifest = fetch(choices[0]["digest"])
    layers = [
        layer
        for layer in manifest.get("layers", [])
        if layer.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    extra_weights = {
        "application/vnd.ollama.image.projector",
        "application/vnd.ollama.image.adapter",
        "application/vnd.ollama.image.embed",
        "application/vnd.ollama.image.tensor",
        "application/vnd.ollama.image.draft",
    }
    if any(
        layer.get("mediaType") in extra_weights for layer in manifest.get("layers", [])
    ):
        raise ValueError("Ollama model needs extra weight files not supported yet")
    if len(layers) != 1:
        raise ValueError("Ollama model needs exactly one GGUF weight layer")
    layer = layers[0]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", layer.get("digest", "")):
        raise ValueError("Invalid Ollama model digest")
    return layer


def hf_model_files(spec: dict, token: str | None) -> tuple[list[str], int]:
    from huggingface_hub import model_info

    info = model_info(
        spec["name"],
        files_metadata=True,
        token=token,
        revision=spec.get("revision") or None,
    )
    files = {item.rfilename: item.size for item in info.siblings}
    if spec["backend"] == "llama.cpp":
        choices = [name for name in files if GGUF_FILE.fullmatch(name)]
        filename = spec.get("filename") or ""
        if not filename:
            preferred = [name for name in choices if "Q4_K_M" in name.upper()]
            selected = preferred if len(preferred) == 1 else choices
            if len(selected) != 1:
                raise ValueError(
                    "Choose a GGUF file with --file; available: "
                    + ", ".join(choices[:20])
                )
            filename = selected[0]
        if filename not in choices:
            raise ValueError(f"GGUF file {filename!r} is absent from {spec['name']}")
        spec["filename"] = filename
        spec["ref"] = f"hf:{spec['name']}@{filename}"
        spec["relative_path"] = f"gguf/hf/{spec['name']}/{filename}"
        return [filename], files[filename] or 0
    weights = [name for name in files if name.endswith(".safetensors")]
    if not weights:
        weights = [name for name in files if name.endswith(".bin")]
    if not weights:
        weights = [name for name in files if name.endswith(".pt")]
    if not weights:
        if spec.get("auto_backend") and any(
            name.lower().endswith(".gguf") for name in files
        ):
            spec["backend"] = "llama.cpp"
            return hf_model_files(spec, token)
        raise ValueError("No vLLM checkpoint weights found in this repository")
    return weights, sum(files[name] or 0 for name in weights)


def model_metadata(spec: dict, weight_bytes: int) -> dict:
    return {
        "ref": spec["ref"],
        "backend": spec["backend"],
        "source": spec["source"],
        "relative_path": spec["relative_path"],
        "weight_bytes": weight_bytes,
        "revision": spec.get("revision") or "main",
    }


def download_model(
    spec: dict,
    root: Path,
    token: str | None,
    max_weight_bytes: int,
    report,
) -> dict:
    """Download on the leader; write a ready marker only after validation."""
    report("Checking model metadata", 0, 0)
    if spec["source"] == "ollama":
        layer = ollama_manifest(spec)
        weight_bytes = int(layer["size"])
        weight_files = []
    else:
        weight_files, weight_bytes = hf_model_files(spec, token)
    if not weight_bytes:
        raise ValueError("Model size is unknown; cannot check GPU capacity")
    if weight_bytes > max_weight_bytes:
        raise ValueError(
            f"Model weights need {weight_bytes / 2**30:.1f} GiB; "
            f"online GPUs allow about {max_weight_bytes / 2**30:.1f} GiB"
        )
    path = root / spec["relative_path"]
    ready = (
        path / ".lanllm-ready"
        if spec["backend"] == "vllm"
        else path.with_name(path.name + ".lanllm-ready")
    )
    if ready.exists() and path.exists():
        try:
            marker = json.loads(ready.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            marker = {}
        if marker.get("revision", "main") == (spec.get("revision") or "main"):
            report("Already downloaded", weight_bytes, weight_bytes)
            return model_metadata(spec, weight_bytes)
    report("Downloading", 0, weight_bytes)
    if spec["source"] == "ollama":
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".gguf.part")
        digest = hashlib.sha256()
        url = f"{REGISTRY}/{spec['repository']}/blobs/{layer['digest']}"
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as output,
            ):
                current = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    current += len(chunk)
                    report("Downloading", current, weight_bytes)
            if (
                current != weight_bytes
                or digest.hexdigest() != layer["digest"].split(":", 1)[1]
            ):
                raise ValueError("Ollama download checksum or size did not match")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(path)
    elif spec["backend"] == "llama.cpp":
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            spec["name"],
            spec["filename"],
            local_dir=path.parent,
            token=token,
            revision=spec.get("revision") or None,
        )
    else:
        from huggingface_hub import snapshot_download

        path.mkdir(parents=True, exist_ok=True)
        extension = Path(weight_files[0]).suffix
        ignored = [
            f"*{other}"
            for other in (".safetensors", ".bin", ".pt", ".gguf")
            if other != extension
        ]
        snapshot_download(
            repo_id=spec["name"],
            local_dir=path,
            token=token,
            revision=spec.get("revision") or None,
            ignore_patterns=ignored,
        )
        if not (path / "config.json").is_file():
            raise ValueError("Hugging Face model has no config.json for vLLM")
    if not path.exists():
        raise ValueError("Downloaded model file is missing")
    ready.write_text(
        json.dumps(model_metadata(spec, weight_bytes)) + "\n", encoding="utf-8"
    )
    report("Ready", weight_bytes, weight_bytes)
    return model_metadata(spec, weight_bytes)
