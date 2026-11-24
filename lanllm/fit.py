"""Choose a model variant using the GPUs currently available to LanLLM."""

from __future__ import annotations

from .model_sources import GGUF_FILE, ollama_manifest, parse_source


def _model_ggufs(files: dict[str, int | None]) -> list[tuple[str, int]]:
    excluded = ("mmproj", "projector", "fastmtp", "draft", "adapter")
    return [
        (name, size)
        for name, size in files.items()
        if GGUF_FILE.fullmatch(name)
        and size is not None
        and size > 0
        and not any(word in name.lower() for word in excluded)
        and "-of-" not in name.lower()
    ]


def choose_fit(
    source: str,
    backend: str | None,
    filename: str,
    leader_bytes: int,
    online_bytes: int,
    token: str | None = None,
) -> dict:
    """Return ranked variants; a size estimate cannot guarantee runtime fit."""
    spec = parse_source(source, backend, filename)
    variants: list[dict] = []
    if spec["source"] == "ollama":
        layer = ollama_manifest(spec)
        variants.append(
            {
                "ref": spec["ref"],
                "source": spec["ref"],
                "backend": "llama.cpp",
                "file": "",
                "weight_bytes": int(layer["size"]),
            }
        )
    else:
        from huggingface_hub import model_info

        info = model_info(spec["name"], files_metadata=True, token=token)
        files = {item.rfilename: item.size for item in info.siblings}
        weights = [
            size
            for name, size in files.items()
            if name.endswith(".safetensors") and size is not None
        ]
        if not weights:
            weights = [
                size
                for name, size in files.items()
                if name.endswith((".bin", ".pt")) and size is not None
            ]
        if weights and (backend in (None, "vllm")) and not spec.get("filename"):
            variants.append(
                {
                    "ref": spec["name"],
                    "source": spec["name"],
                    "backend": "vllm",
                    "file": "",
                    "weight_bytes": sum(weights),
                }
            )
        if backend in (None, "llama.cpp") or spec.get("filename"):
            for name, size in _model_ggufs(files):
                if spec.get("filename") and name != spec["filename"]:
                    continue
                variants.append(
                    {
                        "ref": f"hf:{spec['name']}@{name}",
                        "source": spec["name"],
                        "backend": "llama.cpp",
                        "file": name,
                        "weight_bytes": size,
                    }
                )
    if not variants:
        raise ValueError(
            "No supported single-file GGUF or vLLM checkpoint found in this source"
        )

    for item in variants:
        size = item["weight_bytes"]
        item["fit"] = (
            "leader" if size <= leader_bytes else "pool" if size <= online_bytes else "too_large"
        )
        item["headroom_bytes"] = max(online_bytes - size, 0)
        item["tight"] = size > online_bytes * 0.9

    fits = [item for item in variants if item["fit"] != "too_large"]
    comfortable = [item for item in fits if not item["tight"]]
    choices = comfortable or fits
    recommended = max(
        choices,
        key=lambda item: (item["backend"] == "vllm", item["weight_bytes"]),
        default=None,
    )
    variants.sort(
        key=lambda item: (item["backend"] == "vllm", item["weight_bytes"]),
        reverse=True,
    )
    return {
        "source": source,
        "leader_usable_bytes": leader_bytes,
        "online_usable_bytes": online_bytes,
        "variants": variants,
        "recommended": recommended,
        "method": "weight estimate with 2 GiB/GPU already reserved; 10% additional headroom preferred",
    }
