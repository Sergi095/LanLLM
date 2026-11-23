"""Start a Ray GPU pool and serve any model supported by vLLM."""

import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class Settings:
    role: Literal["head", "worker"]
    node_ip: str
    head_ip: str
    network_interface: str
    local_gpus: int
    total_gpus: int
    model_id: str
    ray_port: int
    api_host: str
    api_port: int
    max_model_len: int
    gpu_memory_utilization: float
    attention_backend: str
    enforce_eager: bool
    hf_revision: str
    quantization: str
    dtype: str
    layer_partition: str
    trust_remote_code: bool
    pool_wait_timeout: int

    @classmethod
    def from_environment(
        cls, role: Literal["head", "worker"], env: dict[str, str]
    ) -> "Settings":
        def required(name: str) -> str:
            value = env.get(name, "").strip()
            if not value:
                raise ValueError(f"{name} must be set")
            return value

        def positive(name: str, default: str) -> int:
            value = int(env.get(name, default))
            if value < 1:
                raise ValueError(f"{name} must be positive")
            return value

        node_ip = required("NODE_IP")
        head_ip = required("HEAD_IP")
        network_interface = required("NETWORK_INTERFACE")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", network_interface):
            raise ValueError("NETWORK_INTERFACE has an invalid name")

        local_gpus = positive("LOCAL_GPUS", "1")
        total_gpus = positive("TOTAL_GPUS", "3")
        if local_gpus > total_gpus:
            raise ValueError("LOCAL_GPUS cannot exceed TOTAL_GPUS")
        utilization = float(env.get("GPU_MEMORY_UTILIZATION", "0.7"))
        if not 0 < utilization <= 1:
            raise ValueError("GPU_MEMORY_UTILIZATION must be in (0, 1]")
        partition = env.get("LAYER_PARTITION", "").strip()
        if partition:
            try:
                layers = [int(part) for part in partition.split(",")]
            except ValueError as exc:
                raise ValueError(
                    "LAYER_PARTITION must be comma-separated integers"
                ) from exc
            if len(layers) != total_gpus or any(layer < 1 for layer in layers):
                raise ValueError("LAYER_PARTITION needs one positive count per GPU")

        return cls(
            role=role,
            node_ip=node_ip,
            head_ip=head_ip,
            network_interface=network_interface,
            local_gpus=local_gpus,
            total_gpus=total_gpus,
            model_id=required("MODEL_ID"),
            ray_port=positive("RAY_PORT", "6379"),
            api_host=env.get("API_HOST", "127.0.0.1"),
            api_port=positive("API_PORT", "8000"),
            max_model_len=positive("MAX_MODEL_LEN", "1024"),
            gpu_memory_utilization=utilization,
            attention_backend=env.get("ATTENTION_BACKEND", "FLASH_ATTN"),
            enforce_eager=env.get("ENFORCE_EAGER", "1") == "1",
            hf_revision=env.get("HF_REVISION", ""),
            quantization=env.get("QUANTIZATION", ""),
            dtype=env.get("DTYPE", ""),
            layer_partition=partition,
            trust_remote_code=env.get("TRUST_REMOTE_CODE", "0") == "1",
            pool_wait_timeout=positive("POOL_WAIT_TIMEOUT", "600"),
        )


def ray_command(settings: Settings) -> list[str]:
    command = ["ray", "start"]
    if settings.role == "head":
        command += [
            "--head",
            f"--port={settings.ray_port}",
            "--include-dashboard=false",
        ]
    else:
        command += [f"--address={settings.head_ip}:{settings.ray_port}", "--block"]
    command += [
        f"--node-ip-address={settings.node_ip}",
        f"--num-gpus={settings.local_gpus}",
        "--disable-usage-stats",
    ]
    return command


def vllm_command(settings: Settings) -> list[str]:
    command = [
        "vllm",
        "serve",
        settings.model_id,
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        str(settings.total_gpus),
        "--distributed-executor-backend",
        "ray",
        "--host",
        settings.api_host,
        "--port",
        str(settings.api_port),
        "--max-model-len",
        str(settings.max_model_len),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--attention-backend",
        settings.attention_backend,
        "--disable-flashinfer-autotune",
    ]
    if settings.enforce_eager:
        command.append("--enforce-eager")
    for flag, value in (
        ("--revision", settings.hf_revision),
        ("--quantization", settings.quantization),
        ("--dtype", settings.dtype),
    ):
        if value:
            command += [flag, value]
    if settings.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def runtime_environment(settings: Settings) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        VLLM_HOST_IP=settings.node_ip,
        NCCL_SOCKET_IFNAME=settings.network_interface,
        GLOO_SOCKET_IFNAME=settings.network_interface,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        VLLM_USE_FLASHINFER_SAMPLER="0",
    )
    if settings.layer_partition:
        env["VLLM_PP_LAYER_PARTITION"] = settings.layer_partition
    return env


def wait_for_gpu_pool(
    settings: Settings, competing_leader: Callable[[], bool] | None = None
) -> bool:
    import ray

    ray.init(address=f"{settings.head_ip}:{settings.ray_port}", logging_level="ERROR")
    deadline = time.monotonic() + settings.pool_wait_timeout
    try:
        while True:
            available = int(ray.cluster_resources().get("GPU", 0))
            if available >= settings.total_gpus:
                print(f"Ray pool ready: {available} GPUs", flush=True)
                return True
            if competing_leader is not None and competing_leader():
                return False
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Ray found {available}/{settings.total_gpus} GPUs before timeout"
                )
            print(f"Waiting for GPUs: {available}/{settings.total_gpus}", flush=True)
            time.sleep(5)
    finally:
        ray.shutdown()
