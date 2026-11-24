"""Discover the LAN pool, elect a Ray head, and expose one API on every node."""

import argparse
import csv
import fcntl
import http.server
import json
import os
import re
import shutil
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from lanllm.model_sources import download_model, model_metadata, parse_source
from lanllm.metrics import SystemSampler
from lanllm.launch import (
    Settings,
    ray_command,
    runtime_environment,
    vllm_command,
    wait_for_gpu_pool,
)


MODEL_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?\Z"
)
REGISTRY_LOCK = threading.Lock()
DOWNLOAD_LOCK = threading.Lock()
SELECTION_LOCK = threading.Lock()
DOWNLOAD_JOB: dict = {"status": "idle"}


def selected_model(root: Path = Path("/models")) -> str | None:
    selection = root / ".lanllm-model"
    try:
        model_id = selection.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not MODEL_ID_PATTERN.fullmatch(model_id) or ".." in model_id:
        raise ValueError(f"Invalid model ID in {selection}: {model_id!r}")
    return model_id


def selected_spec(root: Path = Path("/models")) -> dict | None:
    path = root / ".lanllm-selection.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("backend") not in {"vllm", "llama.cpp"}:
            raise ValueError("Invalid selected backend")
        relative = data.get("relative_path", "")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("Invalid selected model path")
        return data
    legacy = selected_model(root)
    return model_metadata(parse_source(legacy), 0) if legacy else None


def local_ip_for(peer_ip: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((peer_ip, 9))
        return sock.getsockname()[0]


def interface_for(ip: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            try:
                address = fcntl.ioctl(
                    sock.fileno(), 0x8915, struct.pack("256s", name.encode()[:15])
                )[20:24]
            except OSError:
                continue
            if socket.inet_ntoa(address) == ip:
                return name
    raise ValueError(f"No network interface owns {ip}")


def gpu_rows() -> list[dict[str, str]]:
    fields = (
        "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw"
    )
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    keys = ("index", "name", "total_mib", "used_mib", "util", "temp", "watts")
    return [
        dict(zip(keys, (cell.strip() for cell in row), strict=True))
        for row in csv.reader(result.stdout.splitlines())
    ]


def ram_gib() -> tuple[float, float]:
    values = {}
    with open("/proc/meminfo", encoding="ascii") as source:
        for line in source:
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(value.split()[0]) / 1024 / 1024
    return values["MemTotal"], values["MemAvailable"]


@dataclass(frozen=True)
class Config:
    node_ip: str
    peer_ips: tuple[str, ...]
    network_interface: str
    local_gpus: int
    total_gpus: int
    model_id: str
    store_dir: str
    ray_port: int
    api_port: int
    control_port: int
    env: dict[str, str]
    backend: str = "vllm"
    model_file: str = ""

    @property
    def model_path(self) -> Path:
        return (
            Path("/models") / self.model_file
            if self.model_file
            else Path("/models/models") / self.model_id
        )

    @classmethod
    def from_environment(cls, env: dict[str, str]) -> "Config":
        peers = tuple(
            part.strip() for part in env.get("PEER_IPS", "").split(",") if part.strip()
        )
        if not peers:
            raise ValueError("PEER_IPS must list every LAN node, including this one")
        if len(set(peers)) != len(peers):
            raise ValueError("PEER_IPS contains duplicate addresses")
        node_ip = env.get("NODE_IP", "").strip() or local_ip_for(peers[0])
        if node_ip not in peers:
            raise ValueError(
                f"Detected NODE_IP={node_ip}, which is absent from PEER_IPS"
            )
        interface = env.get("NETWORK_INTERFACE", "").strip() or interface_for(node_ip)
        rows = gpu_rows()
        if not rows:
            raise ValueError("No NVIDIA GPUs are visible inside the container")
        local_gpus = len(rows)
        total_gpus = int(env.get("TOTAL_GPUS", "3"))
        if total_gpus < local_gpus:
            raise ValueError("TOTAL_GPUS is less than this node's GPU count")
        model_id = env.get("MODEL_ID", "").strip()
        if not model_id:
            raise ValueError("MODEL_ID must be set")
        if not MODEL_ID_PATTERN.fullmatch(model_id) or ".." in model_id:
            raise ValueError("MODEL_ID must be a Hugging Face model ID")
        store_dir = env.get("MODEL_STORE_DIR", "/srv/lanllm-store").strip()
        if not Path(store_dir).is_absolute():
            raise ValueError("MODEL_STORE_DIR must be an absolute path")
        return cls(
            node_ip=node_ip,
            peer_ips=peers,
            network_interface=interface,
            local_gpus=local_gpus,
            total_gpus=total_gpus,
            model_id=model_id,
            store_dir=store_dir,
            ray_port=int(env.get("RAY_PORT", "6379")),
            api_port=int(env.get("API_PORT", "8001")),
            control_port=int(env.get("CONTROL_PORT", "8501")),
            env=env,
        )

    def settings(self, role: str, leader_ip: str, partition: str = "") -> Settings:
        values = dict(self.env)
        values.update(
            NODE_IP=self.node_ip,
            HEAD_IP=leader_ip,
            MODEL_ID=self.model_id,
            NETWORK_INTERFACE=self.network_interface,
            LOCAL_GPUS=str(self.local_gpus),
            TOTAL_GPUS=str(self.total_gpus),
            API_HOST="0.0.0.0",
            API_PORT=str(self.api_port),
            RAY_PORT=str(self.ray_port),
            LAYER_PARTITION=partition,
            MODEL_SOURCE=str(self.model_path),
        )
        return Settings.from_environment(role, values)


class State:
    def __init__(self, config: Config):
        self.config = config
        self.role = "candidate"
        self.leader_ip = ""
        self.lock = threading.Lock()
        self.system_sampler = SystemSampler(config.network_interface)

    def set(self, role: str, leader_ip: str) -> None:
        with self.lock:
            self.role = role
            self.leader_ip = leader_ip
        print(
            f"LanLLM: {role} on {self.config.node_ip}; leader {leader_ip}", flush=True
        )

    def identity(self) -> dict:
        with self.lock:
            role, leader_ip = self.role, self.leader_ip
        local_store = Path("/models-local")
        has_model = (
            local_store / self.config.model_file
            if self.config.model_file
            else local_store / "models" / self.config.model_id / ".lanllm-ready"
        ).exists()
        has_selection = any(
            (local_store / name).is_file()
            for name in (".lanllm-model", ".lanllm-selection.json")
        )
        return {
            "node_ip": self.config.node_ip,
            "role": role,
            "leader_ip": leader_ip,
            "store_priority": int(has_model or has_selection),
            "model_store_dir": self.config.store_dir,
        }

    def snapshot(self) -> dict:
        total_ram, available_ram = ram_gib()
        return self.identity() | {
            "model_id": self.config.model_id,
            "backend": self.config.backend,
            "gpus": gpu_rows(),
            "cpus": os.cpu_count(),
            "ram_total_gib": round(total_ram, 1),
            "ram_available_gib": round(available_ram, 1),
            **self.system_sampler.sample(),
        }


class ControlHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            payload = json.dumps(self.server.state.identity()).encode()
        elif self.path == "/state":
            payload = json.dumps(self.server.state.snapshot()).encode()
        elif self.path in {"/nodes", "/models"}:
            if self.server.state.identity()["role"] != "leader":
                self.send_error(409, "Query the elected leader")
                return
            nodes = registered_nodes(self.server.state.config, self.server.state)
            data = (
                {"nodes": nodes}
                if self.path == "/nodes"
                else model_catalog(self.server.state.config, nodes)
            )
            payload = json.dumps(data).encode()
        elif self.path == "/download":
            with DOWNLOAD_LOCK:
                payload = json.dumps(DOWNLOAD_JOB).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path not in {"/model", "/download", "/delete", "/fit"}:
            self.send_error(404)
            return
        if self.client_address[0] not in self.server.state.config.peer_ips:
            self.send_error(403)
            return
        if self.server.state.identity()["role"] != "leader":
            self.send_error(409, "This node is not the leader")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048:
                raise ValueError("Invalid request length")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
        except (ValueError, TypeError) as exc:
            self.send_error(400, str(exc))
            return
        if self.path == "/fit":
            from .fit import choose_fit

            try:
                config = self.server.state.config
                nodes = registered_nodes(config, self.server.state)
                catalog = model_catalog(config, nodes)
                plan = choose_fit(
                    data["source"],
                    data.get("backend"),
                    data.get("file", ""),
                    catalog["leader_usable_mib"] * 1048576,
                    catalog["online_usable_mib"] * 1048576,
                    config.env.get("HF_TOKEN") or None,
                )
                plan["online_gpus"] = sum(
                    len(node["gpus"]) for node in nodes if node["online"]
                )
                plan["required_gpus"] = config.total_gpus
            except Exception as exc:
                self.reply_json({"error": str(exc)}, status=400)
                return
            self.reply_json(plan)
            return
        if self.path == "/download":
            try:
                spec = parse_source(
                    data["source"], data.get("backend"), data.get("file", "")
                )
                start_download(spec, self.server.state)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_error(409, str(exc))
                return
            self.reply_json({"status": "started", "ref": spec["ref"]})
            return
        if self.path == "/delete":
            try:
                model_id = data["model_id"]
                if not isinstance(model_id, str):
                    raise ValueError("Model reference must be a string")
                config = self.server.state.config
                catalog = model_catalog(
                    config, registered_nodes(config, self.server.state)
                )
                matches = [
                    item for item in catalog["models"] if item["model_id"] == model_id
                ]
                if len(matches) != 1:
                    raise ValueError("Model is not available in the leader's library")
                with SELECTION_LOCK:
                    selection = selected_spec(Path("/models"))
                    active_refs = {config.model_id}
                    if selection:
                        active_refs.add(selection["ref"])
                    freed = delete_downloaded_model(matches[0], active_refs)
            except (ValueError, KeyError, OSError) as exc:
                self.reply_json({"error": str(exc)}, status=409)
                return
            self.reply_json({"model_id": model_id, "weight_bytes": freed})
            return
        config = self.server.state.config
        online_gpus = config.local_gpus
        for ip in config.peer_ips:
            if ip == config.node_ip:
                continue
            peer = query_node(ip, config.control_port, full=True)
            if peer is not None:
                online_gpus += len(peer["gpus"])
        try:
            model_id = data["model_id"]
            if not isinstance(model_id, str):
                raise ValueError("Invalid model reference")
            if model_id.startswith(("ollama:", "hf:")):
                catalog = model_catalog(
                    config, registered_nodes(config, self.server.state)
                )
                matches = [
                    item for item in catalog["models"] if item["model_id"] == model_id
                ]
                if len(matches) != 1:
                    raise ValueError("Model is not available in the leader's library")
                fit = matches[0]["fit"]
                if fit == "too_large":
                    raise ValueError("Selected model exceeds the registered GPU pool capacity")
                selection = {
                    key: matches[0][key]
                    for key in (
                        "ref",
                        "backend",
                        "source",
                        "relative_path",
                        "weight_bytes",
                    )
                }
            else:
                fit = "unknown"
                selection = model_metadata(parse_source(model_id), 0)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_error(400, str(exc))
            return
        required_gpus = config.local_gpus if fit == "leader" else config.total_gpus
        if online_gpus < required_gpus:
            self.send_error(
                409,
                f"Only {online_gpus}/{required_gpus} GPUs are online",
            )
            return
        temporary = (
            Path("/models") / f".lanllm-selection.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with SELECTION_LOCK:
                temporary.write_text(json.dumps(selection) + "\n", encoding="utf-8")
                temporary.replace(Path("/models/.lanllm-selection.json"))
        except OSError as exc:
            self.send_error(503, str(exc))
            return
        self.reply_json({"model_id": model_id})

    def reply_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class ControlServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, state: State):
        self.state = state
        super().__init__(
            (state.config.node_ip, state.config.control_port), ControlHandler
        )


def query_node(
    ip: str, port: int, timeout: float = 1, *, full: bool = False
) -> dict | None:
    path = "state" if full else "health"
    try:
        with urllib.request.urlopen(
            f"http://{ip}:{port}/{path}", timeout=timeout
        ) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None


def ip_key(ip: str) -> tuple[int, ...]:
    return tuple(int(part) for part in ip.split("."))


def elect_leader(node_ip: str, peer_states: list[dict], local_priority: int = 0) -> str:
    candidates = peer_states + [
        {"node_ip": node_ip, "role": "candidate", "store_priority": local_priority}
    ]
    return min(
        candidates,
        key=lambda state: (
            -state.get("store_priority", 0),
            state.get("role") != "leader",
            ip_key(state["node_ip"]),
        ),
    )["node_ip"]


def allocate_layers(layer_count: int, memory_mib: list[int]) -> list[int]:
    if layer_count < len(memory_mib) or not memory_mib:
        raise ValueError("Model needs at least one layer per GPU")
    # Leave extra room on small cards for embeddings, runtime, and KV cache.
    weights = [
        max(memory - (3072 if memory < 10240 else 2048), 1024) for memory in memory_mib
    ]
    remaining = layer_count - len(weights)
    raw = [remaining * weight / sum(weights) for weight in weights]
    layers = [1 + int(value) for value in raw]
    extras = layer_count - sum(layers)
    order = sorted(range(len(raw)), key=lambda index: raw[index] % 1, reverse=True)
    for index in order[:extras]:
        layers[index] += 1
    return layers


def resolve_partition(config: Config, leader_ip: str) -> str:
    profiles = json.loads(config.env.get("MODEL_PROFILES_JSON", "{}"))
    profile = profiles.get(config.model_id, {})
    requested = str(profile.get("layer_partition") or config.env.get("LAYER_PARTITION", "")).strip()
    if requested != "auto":
        return requested
    ordered_ips = [leader_ip] + [ip for ip in config.peer_ips if ip != leader_ip]
    while True:
        states = [query_node(ip, config.control_port, full=True) for ip in ordered_ips]
        if all(state is not None for state in states):
            memory = [
                int(gpu["total_mib"]) for state in states for gpu in state["gpus"]
            ]
            if len(memory) == config.total_gpus:
                break
        print("Waiting for GPU inventory to calculate layer partition", flush=True)
        time.sleep(5)
    with open(config.model_path / "config.json", encoding="utf-8") as source:
        model_config = json.load(source)
    layer_count = model_layer_count(model_config)
    if not isinstance(layer_count, int):
        raise ValueError(
            "Auto layer partition requires num_hidden_layers in config.json"
        )
    partition = ",".join(str(count) for count in allocate_layers(layer_count, memory))
    print(f"Auto layer partition: {partition} for GPUs {memory} MiB", flush=True)
    return partition


def model_layer_count(model_config: dict) -> int | None:
    count = model_config.get("num_hidden_layers")
    if count is None and isinstance(model_config.get("text_config"), dict):
        count = model_config["text_config"].get("num_hidden_layers")
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def verify_model_store(root: Path = Path("/models")) -> None:
    marker = root / ".lanllm-store"
    try:
        valid = marker.read_text(encoding="ascii").strip() == "lanllm-model-store-v1"
    except OSError:
        valid = False
    if not valid:
        raise ValueError(
            f"The LanLLM model store is unavailable at {root}; "
            "check the local folder or leader NFS mount"
        )


def registered_nodes(
    config: Config, state: State, root: Path = Path("/models")
) -> list[dict]:
    path = root / ".lanllm-nodes.json"
    with REGISTRY_LOCK:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))["nodes"]
            if not isinstance(saved, dict):
                saved = {}
        except (OSError, ValueError, KeyError, TypeError):
            saved = {}
        updated = dict(saved)
        online = {config.node_ip: state.snapshot()}
        for ip in config.peer_ips:
            if ip == config.node_ip:
                continue
            peer = query_node(ip, config.control_port, timeout=2, full=True)
            if peer is not None:
                online[ip] = peer
        for ip, snapshot in online.items():
            updated[ip] = {
                "gpus": [
                    {
                        "index": gpu["index"],
                        "name": gpu["name"],
                        "total_mib": gpu["total_mib"],
                    }
                    for gpu in snapshot["gpus"]
                ],
                "cpus": snapshot["cpus"],
                "ram_total_gib": snapshot["ram_total_gib"],
            }
        if updated != saved:
            temporary = path.with_name(
                f".lanllm-nodes.{os.getpid()}.{threading.get_ident()}"
            )
            temporary.write_text(
                json.dumps({"version": 1, "nodes": updated}, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        return [
            {
                "node_ip": ip,
                "online": ip in online,
                "role": online[ip]["role"] if ip in online else "offline",
                **updated.get(ip, {"gpus": [], "cpus": None, "ram_total_gib": None}),
            }
            for ip in config.peer_ips
        ]


def estimate_fit(weight_bytes: int, leader_mib: int, pool_mib: int) -> str:
    if weight_bytes <= 0:
        return "unknown"
    weight_mib = (weight_bytes + 1048575) // 1048576
    if weight_mib <= leader_mib:
        return "leader"
    if weight_mib <= pool_mib:
        return "pool"
    return "too_large"


def model_catalog(
    config: Config, nodes: list[dict], library_root: Path = Path("/models")
) -> dict:
    def budget(selected: list[dict]) -> int:
        return sum(
            max(
                int(int(gpu["total_mib"]) * utilization) - 2048,
                0,
            )
            for node in selected
            for gpu in node["gpus"]
        )

    utilization = float(config.env.get("GPU_MEMORY_UTILIZATION", "0.7"))
    leader = [node for node in nodes if node["node_ip"] == config.node_ip]
    leader_mib = budget(leader)
    pool_mib = budget(nodes)
    online_mib = budget([node for node in nodes if node["online"]])
    models = []
    root = library_root / "models"
    if root.is_dir():
        candidates = [
            candidate
            for owner in root.iterdir()
            if owner.is_dir()
            for candidate in [owner, *owner.iterdir()]
            if candidate.is_dir() and (candidate / ".lanllm-ready").is_file()
        ]
        for directory in sorted(candidates):
            weights = [
                file
                for file in directory.rglob("*")
                if file.is_file() and file.suffix in {".safetensors", ".bin", ".pt"}
            ]
            weight_bytes = sum(file.stat().st_size for file in weights)
            weight_mib = (weight_bytes + 1048575) // 1048576
            models.append(
                {
                    "model_id": directory.relative_to(root).as_posix(),
                    "ref": directory.relative_to(root).as_posix(),
                    "backend": "vllm",
                    "source": "hf",
                    "relative_path": f"models/{directory.relative_to(root).as_posix()}",
                    "weight_bytes": weight_bytes,
                    "fit": estimate_fit(weight_bytes, leader_mib, pool_mib),
                    "fits_online_vram": bool(weights) and weight_mib <= online_mib,
                }
            )
    gguf_root = library_root / "gguf"
    if gguf_root.is_dir():
        for marker in sorted(gguf_root.rglob("*.gguf.lanllm-ready")):
            try:
                metadata = json.loads(marker.read_text(encoding="utf-8"))
                relative = metadata["relative_path"]
                file = library_root / relative
                if (
                    metadata["backend"] != "llama.cpp"
                    or not file.is_file()
                    or not file.resolve().is_relative_to(gguf_root.resolve())
                ):
                    continue
                weight_bytes = file.stat().st_size
                weight_mib = (weight_bytes + 1048575) // 1048576
                models.append(
                    {
                        "model_id": metadata["ref"],
                        **metadata,
                        "weight_bytes": weight_bytes,
                        "fit": estimate_fit(weight_bytes, leader_mib, pool_mib),
                        "fits_online_vram": weight_mib <= online_mib,
                    }
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return {
        "models": models,
        "leader_usable_mib": leader_mib,
        "registered_usable_mib": pool_mib,
        "online_usable_mib": online_mib,
        "registered_nodes": sum(bool(node["gpus"]) for node in nodes),
        "online_nodes": sum(node["online"] for node in nodes),
    }


def selected_model_fit(catalog: dict, model_id: str) -> str:
    return next(
        (item["fit"] for item in catalog["models"] if item["model_id"] == model_id),
        "unknown",
    )


def online_gpu_count(nodes: list[dict]) -> int:
    return sum(len(node["gpus"]) for node in nodes if node["online"])


def delete_downloaded_model(
    entry: dict, active_refs: set[str], library_root: Path = Path("/models")
) -> int:
    """Remove one catalogued model from the leader's shared library."""
    global DOWNLOAD_JOB
    ref = entry["model_id"]
    if ref in active_refs:
        raise ValueError("Switch to another model before deleting the active model")
    spec = parse_source(ref)
    relative = entry["relative_path"]
    if (
        spec["relative_path"] != relative
        or spec["backend"] != entry["backend"]
    ):
        raise ValueError("Model catalog path does not match its reference")
    path = library_root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(library_root.resolve()):
        raise ValueError("Model path is outside the shared library")
    marker = (
        path / ".lanllm-ready" if entry["backend"] == "vllm"
        else path.with_name(path.name + ".lanllm-ready")
    )
    if not marker.is_file() or not path.exists():
        raise ValueError("Model files are missing or incomplete")
    with DOWNLOAD_LOCK:
        if DOWNLOAD_JOB.get("status") in {"checking", "downloading", "deleting"}:
            raise ValueError("Wait for the current model library operation to finish")
        DOWNLOAD_JOB = {"status": "deleting", "ref": ref}
    try:
        if entry["backend"] == "vllm":
            if not path.is_dir():
                raise ValueError("Model directory is missing")
            shutil.rmtree(path)
        else:
            if not path.is_file():
                raise ValueError("GGUF model file is missing")
            path.unlink()
            marker.unlink()
            if entry["source"] == "hf":
                cache = path.parent / ".cache/huggingface/download"
                for suffix in (".metadata", ".lock"):
                    (cache / f"{path.name}{suffix}").unlink(missing_ok=True)
        store_root = library_root / (
            "models" if entry["backend"] == "vllm" else "gguf"
        )
        directory = path.parent
        while directory != store_root:
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent
        return int(entry["weight_bytes"])
    finally:
        with DOWNLOAD_LOCK:
            DOWNLOAD_JOB = {"status": "idle"}


def start_download(spec: dict, state: State) -> None:
    global DOWNLOAD_JOB
    with DOWNLOAD_LOCK:
        if DOWNLOAD_JOB.get("status") in {"checking", "downloading", "deleting"}:
            raise ValueError("Another model library operation is already running")
        DOWNLOAD_JOB = {
            "status": "checking",
            "stage": "Checking model metadata",
            "ref": spec["ref"],
            "downloaded_bytes": 0,
            "total_bytes": 0,
        }

    def report(stage: str, current: int, total: int) -> None:
        with DOWNLOAD_LOCK:
            DOWNLOAD_JOB.update(
                stage=stage,
                status="downloading" if stage == "Downloading" else "checking",
                ref=spec["ref"],
                downloaded_bytes=current,
                total_bytes=total,
            )

    def work() -> None:
        global DOWNLOAD_JOB
        try:
            config = state.config
            spec["revision"] = (
                config.env.get("HF_REVISION") or "main"
                if spec["source"] == "hf"
                else "main"
            )
            nodes = registered_nodes(config, state)
            online_mib = model_catalog(config, nodes)["online_usable_mib"]
            metadata = download_model(
                spec,
                Path("/models"),
                config.env.get("HF_TOKEN") or None,
                online_mib * 1048576,
                report,
            )
            with DOWNLOAD_LOCK:
                DOWNLOAD_JOB.update(status="ready", stage="Ready", ref=metadata["ref"])
        except Exception as exc:
            with DOWNLOAD_LOCK:
                DOWNLOAD_JOB.update(status="error", stage="Failed", error=str(exc))

    threading.Thread(target=work, daemon=True).start()


def mount_model_store(config: Config, leader_ip: str) -> None:
    target = Path("/models")
    if target.is_mount():
        raise RuntimeError(f"Model store is already mounted at {target}")
    if leader_ip == config.node_ip:
        command = ["mount", "--bind", "/models-local", str(target)]
    else:
        leader = query_node(leader_ip, config.control_port)
        if leader is None or leader.get("role") != "leader":
            raise RuntimeError(f"Leader {leader_ip} is not ready to export models")
        export_dir = leader.get("model_store_dir", "")
        if not Path(export_dir).is_absolute():
            raise RuntimeError(f"Leader {leader_ip} advertised an invalid model path")
        command = [
            "mount",
            "-t",
            "nfs4",
            "-o",
            "ro,vers=4.2,soft,timeo=50,retrans=2",
            f"{leader_ip}:{export_dir}",
            str(target),
        ]
    subprocess.run(command, check=True, timeout=45)
    try:
        verify_model_store(target)
    except Exception:
        unmount_model_store()
        raise
    print(f"Model store mounted from {leader_ip} at {target}", flush=True)


def unmount_model_store() -> None:
    if Path("/models").is_mount():
        subprocess.run(["umount", "-l", "/models"], check=False, timeout=15)


def model_ready(config: Config) -> bool:
    if config.backend == "llama.cpp":
        return (
            config.model_path.is_file()
            and config.model_path.with_name(
                config.model_path.name + ".lanllm-ready"
            ).is_file()
        )
    revision = config.env.get("HF_REVISION", "").strip() or "main"
    try:
        marker = (config.model_path / ".lanllm-ready").read_text().strip()
        if marker.startswith("{"):
            marker = json.loads(marker).get("revision", "main")
        return (config.model_path / "config.json").is_file() and (marker == revision)
    except (OSError, ValueError):
        return False


def ensure_model(config: Config) -> None:
    if model_ready(config):
        print(f"Using leader model store at {config.model_path}", flush=True)
        return
    from huggingface_hub import snapshot_download

    config.model_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.model_id} into {config.model_path}", flush=True)
    snapshot_download(
        repo_id=config.model_id,
        local_dir=config.model_path,
        revision=config.env.get("HF_REVISION") or None,
        token=config.env.get("HF_TOKEN") or None,
    )
    if not (config.model_path / "config.json").is_file():
        raise RuntimeError("The model download has no config.json")
    revision = config.env.get("HF_REVISION", "").strip() or "main"
    temporary = config.model_path / f".lanllm-ready.{os.getpid()}"
    temporary.write_text(revision + "\n", encoding="ascii")
    temporary.replace(config.model_path / ".lanllm-ready")
    print(f"Leader model ready: {config.model_path}", flush=True)


def wait_for_model(config: Config, leader_ip: str) -> bool:
    failures = 0
    while not model_ready(config):
        leader = query_node(leader_ip, config.control_port)
        if leader is None or leader.get("role") != "leader":
            failures += 1
            if failures >= 3:
                return False
        else:
            failures = 0
        print(f"Waiting for leader model {config.model_id}", flush=True)
        time.sleep(5)
    return True


class ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        if self.server.annotate_models:
            try:
                prefix = self.request.recv(4096, socket.MSG_PEEK)
                if prefix.startswith(b"GET /v1/models "):
                    self.serve_models()
                    return
            except OSError:
                return
        try:
            upstream = socket.create_connection(
                (self.server.upstream_ip, self.server.upstream_port), timeout=3
            )
        except OSError:
            message = b"LanLLM leader is starting"
            self.request.sendall(
                b"HTTP/1.1 503 Service Unavailable\r\n"
                + f"Content-Length: {len(message)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + message
            )
            return
        with upstream:
            upstream.settimeout(None)

            def copy(source: socket.socket, target: socket.socket) -> None:
                try:
                    while data := source.recv(65536):
                        target.sendall(data)
                except OSError:
                    pass
                try:
                    target.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

            send = threading.Thread(
                target=copy, args=(self.request, upstream), daemon=True
            )
            send.start()
            copy(upstream, self.request)
            send.join(timeout=1)

    def serve_models(self) -> None:
        self.request.settimeout(3)
        request = bytearray()
        while b"\r\n\r\n" not in request and len(request) < 65536:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            request.extend(chunk)
        if b"\r\n\r\n" not in request:
            return
        try:
            with urllib.request.urlopen(
                f"http://{self.server.upstream_ip}:{self.server.upstream_port}/v1/models",
                timeout=5,
            ) as response:
                payload = annotate_model_context(json.load(response))
            body = json.dumps(payload).encode()
            self.request.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + body
            )
        except (OSError, ValueError, urllib.error.URLError):
            body = b"LanLLM leader is starting"
            self.request.sendall(
                b"HTTP/1.1 503 Service Unavailable\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + body
            )


def annotate_model_context(payload: dict) -> dict:
    """Advertise vLLM context in the field used by local-model clients."""
    for model in payload.get("data", []):
        length = model.get("max_model_len")
        if isinstance(length, int) and length > 0:
            model.setdefault("meta", {})["n_ctx"] = length
    return payload


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self, config: Config, upstream_ip: str, upstream_port: int | None = None,
        annotate_models: bool = False,
    ):
        self.upstream_ip = upstream_ip
        self.upstream_port = upstream_port or config.api_port
        self.annotate_models = annotate_models
        super().__init__(("0.0.0.0", config.api_port), ProxyHandler)


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def stop_ray() -> None:
    subprocess.run(["ray", "stop", "--force"], check=False, timeout=30)


def reachable(ip: str, port: int) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=1):
            return True
    except OSError:
        return False


def llama_rpc_command(config: Config) -> list[str]:
    return [
        "ggml-rpc-server",
        "--host",
        config.node_ip,
        "--port",
        config.env.get("LLAMA_RPC_PORT", "50052"),
    ]


def llama_server_command(config: Config, endpoints: list[str]) -> list[str]:
    command = [
        "llama-server",
        "--model",
        str(config.model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(config.api_port),
        "--alias",
        config.model_id,
        "--ctx-size",
        config.env.get("MAX_MODEL_LEN", "2048"),
        "--parallel",
        config.env.get("LLAMA_PARALLEL", "1"),
        "--n-gpu-layers",
        "all",
    ]
    if endpoints:
        command += ["--rpc", ",".join(endpoints)]
    if split := config.env.get("LLAMA_TENSOR_SPLIT", "").strip():
        command += ["--tensor-split", split]
    for key, flag in (
        ("LLAMA_CACHE_TYPE_K", "--cache-type-k"),
        ("LLAMA_CACHE_TYPE_V", "--cache-type-v"),
        ("LLAMA_BATCH_SIZE", "--batch-size"),
        ("LLAMA_UBATCH_SIZE", "--ubatch-size"),
    ):
        if value := config.env.get(key, "").strip():
            command += [flag, value]
    return command


def lower_priority_leader(config: Config) -> str | None:
    local_store = Path("/models-local")
    local_model = (
        (local_store / config.model_file).exists()
        if config.model_file
        else (local_store / "models" / config.model_id / ".lanllm-ready").is_file()
    )
    local_priority = int(
        (local_store / ".lanllm-model").is_file()
        or (local_store / ".lanllm-selection.json").is_file()
        or local_model
    )
    for ip in config.peer_ips:
        if ip == config.node_ip:
            continue
        peer = query_node(ip, config.control_port)
        if (
            peer is not None
            and peer.get("role") == "leader"
            and (
                peer.get("store_priority", 0) > local_priority
                or (
                    peer.get("store_priority", 0) == local_priority
                    and ip_key(ip) < ip_key(config.node_ip)
                )
            )
        ):
            return ip
    return None


def run_worker(config: Config, state: State, leader_ip: str) -> None:
    state.set("worker", leader_ip)
    proxy = ProxyServer(config, leader_ip)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    process = None
    failures = 0
    try:
        if not wait_for_model(config, leader_ip):
            return
        try:
            with urllib.request.urlopen(
                f"http://{leader_ip}:{config.control_port}/models", timeout=10
            ) as response:
                leader_fit = selected_model_fit(json.load(response), config.model_id)
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            leader_fit = "unknown"
        if leader_fit == "leader":
            print("Selected model runs on the leader's local GPUs", flush=True)
            while True:
                leader = query_node(leader_ip, config.control_port)
                if leader is None or leader.get("role") != "leader":
                    failures += 1
                    if failures >= 3:
                        return
                else:
                    failures = 0
                time.sleep(2)
        if config.backend == "llama.cpp":
            process = subprocess.Popen(llama_rpc_command(config))
            while process.poll() is None:
                leader = query_node(leader_ip, config.control_port)
                if leader is None or leader.get("role") != "leader":
                    failures += 1
                    if failures >= 3:
                        return
                else:
                    failures = 0
                time.sleep(2)
            raise RuntimeError(
                f"llama.cpp RPC server exited with code {process.returncode}"
            )
        settings = config.settings(
            "worker", leader_ip, resolve_partition(config, leader_ip)
        )
        while True:
            leader = query_node(leader_ip, config.control_port)
            if leader is None or leader.get("role") != "leader":
                failures += 1
                if failures >= 3:
                    return
            else:
                failures = 0
            if process is not None and process.poll() is not None:
                print(
                    f"Ray worker exited with code {process.returncode}; retrying",
                    flush=True,
                )
                process = None
            if process is None and reachable(leader_ip, config.ray_port):
                process = subprocess.Popen(
                    ray_command(settings), env=runtime_environment(settings)
                )
            time.sleep(2)
    finally:
        stop_process(process)
        stop_ray()
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=3)


def run_leader(config: Config, state: State) -> None:
    state.set("leader", config.node_ip)
    if lower_priority_leader(config) is not None:
        return
    nodes = registered_nodes(config, state)
    fit = selected_model_fit(model_catalog(config, nodes), config.model_id)
    local_only = fit == "leader"
    target_gpus = config.local_gpus if local_only else config.total_gpus
    print(
        f"Model placement: {'leader' if local_only else 'pool'} "
        f"({target_gpus} GPU(s))",
        flush=True,
    )
    while True:
        peers = [
            peer
            for ip in config.peer_ips
            if ip != config.node_ip
            if (peer := query_node(ip, config.control_port, full=True)) is not None
        ]
        local_priority = state.identity()["store_priority"]
        winner = elect_leader(config.node_ip, peers, local_priority)
        available_gpus = config.local_gpus + sum(len(peer["gpus"]) for peer in peers)
        if winner != config.node_ip:
            return
        if available_gpus >= target_gpus:
            break
        print(
            f"Waiting for GPU inventory: {available_gpus}/{target_gpus}",
            flush=True,
        )
        time.sleep(5)
    registered_nodes(config, state)
    if config.backend == "llama.cpp":
        if not model_ready(config):
            raise RuntimeError(f"Download {config.model_id} before selecting it")
        rpc_port = int(config.env.get("LLAMA_RPC_PORT", "50052"))
        endpoints = (
            [f"{ip}:{rpc_port}" for ip in config.peer_ips if ip != config.node_ip]
            if not local_only else []
        )
        while endpoints and not all(
            reachable(ip, rpc_port) for ip in config.peer_ips if ip != config.node_ip
        ):
            print("Waiting for llama.cpp RPC workers", flush=True)
            time.sleep(3)
        server = subprocess.Popen(llama_server_command(config, endpoints))
        try:
            while server.poll() is None:
                if lower_priority_leader(config) is not None:
                    return
                nodes = registered_nodes(config, state)
                if not local_only and online_gpu_count(nodes) < target_gpus:
                    print("GPU pool lost capacity; stopping model", flush=True)
                    return
                time.sleep(5)
            raise RuntimeError(f"llama.cpp server exited with code {server.returncode}")
        finally:
            stop_process(server)
    ensure_model(config)
    settings = config.settings(
        "head", config.node_ip,
        "" if local_only else resolve_partition(config, config.node_ip),
    )
    if local_only:
        settings = replace(settings, total_gpus=config.local_gpus)
    internal_port = config.api_port + 10000
    if internal_port > 65535:
        raise ValueError("API_PORT must be at most 55535")
    inference_settings = replace(
        settings, api_host="127.0.0.1", api_port=internal_port
    )
    public_proxy = ProxyServer(
        config, "127.0.0.1", internal_port, annotate_models=True
    )
    proxy_thread = threading.Thread(target=public_proxy.serve_forever, daemon=True)
    proxy_thread.start()
    server = None
    try:
        subprocess.run(
            ray_command(settings), check=True, env=runtime_environment(settings)
        )
        while True:
            try:
                ready = wait_for_gpu_pool(
                    settings,
                    competing_leader=lambda: lower_priority_leader(config) is not None,
                )
            except TimeoutError as exc:
                print(f"{exc}; waiting for nodes to join", flush=True)
                continue
            if not ready:
                print("Another node won the leader election", flush=True)
                return
            server = subprocess.Popen(
                vllm_command(inference_settings),
                env=runtime_environment(inference_settings),
            )
            last_registry_update = 0.0
            while server.poll() is None:
                other_leader = lower_priority_leader(config)
                if other_leader is not None:
                    print(f"Another leader won election at {other_leader}", flush=True)
                    return
                if time.monotonic() - last_registry_update >= 15:
                    nodes = registered_nodes(config, state)
                    if not local_only and online_gpu_count(nodes) < target_gpus:
                        print("GPU pool lost capacity; stopping model", flush=True)
                        return
                    last_registry_update = time.monotonic()
                time.sleep(3)
            print(f"vLLM exited with code {server.returncode}; restarting", flush=True)
    finally:
        stop_process(server)
        stop_ray()
        public_proxy.shutdown()
        public_proxy.server_close()
        proxy_thread.join(timeout=3)


def run(config: Config) -> None:
    verify_model_store(Path("/models-local"))
    state = State(config)
    control = ControlServer(state)
    control_thread = threading.Thread(target=control.serve_forever, daemon=True)
    control_thread.start()

    def watch_selection() -> None:
        while True:
            time.sleep(2)
            if not Path("/models").is_mount():
                continue
            try:
                selection = selected_spec()
                changed = selection is not None and (
                    selection["ref"] != state.config.model_id
                    or selection["backend"] != state.config.backend
                )
            except (OSError, ValueError) as exc:
                print(f"Model selection error: {exc}", flush=True)
                continue
            if changed:
                print(
                    "Shared model selection changed; restarting this node", flush=True
                )
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    threading.Thread(target=watch_selection, daemon=True).start()
    print(
        f"LanLLM node {config.node_ip} ({config.network_interface}), "
        f"{config.local_gpus} local GPU(s)",
        flush=True,
    )
    try:
        # Give simultaneously started nodes time to announce candidacy.
        time.sleep(3)
        while True:
            peers = [
                peer
                for ip in config.peer_ips
                if ip != config.node_ip
                if (peer := query_node(ip, config.control_port)) is not None
            ]
            leader_ip = elect_leader(
                config.node_ip, peers, state.identity()["store_priority"]
            )
            try:
                mount_model_store(config, leader_ip)
                selection = selected_spec()
                active = replace(
                    config,
                    model_id=selection["ref"] if selection else config.model_id,
                    backend=selection["backend"] if selection else "vllm",
                    model_file=selection["relative_path"] if selection else "",
                )
                state.config = active
                if leader_ip == config.node_ip:
                    run_leader(active, state)
                else:
                    run_worker(active, state, leader_ip)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                print(f"Model store unavailable from {leader_ip}: {exc}", flush=True)
            finally:
                unmount_model_store()
                state.config = config
            state.set("candidate", "")
            time.sleep(2)
    finally:
        control.shutdown()
        control.server_close()


def format_status(states: list[dict], offline: list[str], api_port: int) -> str:
    lines = ["LanLLM GPU pool", "=" * 109]
    if states:
        leaders = [state for state in states if state["role"] == "leader"]
        if leaders:
            active = leaders[0]
            backend = active.get("backend", "vllm")
            lines.append(
                f"Model: {active['model_id']} ({backend})   Leader: {active['node_ip']}"
            )
        else:
            lines.append("Model: none loaded   Leader: starting")
    lines.append(
        f"{'Node / role':<25} {'GPU':<3} {'Name':<36} {'Temp':>5} "
        f"{'Util':>5} {'VRAM used / total':>21} {'Power':>8}"
    )
    lines.append("-" * 109)
    total_gpus = total_mib = used_mib = 0
    for state in states:
        for gpu in state["gpus"]:
            total_gpus += 1
            total_mib += int(gpu["total_mib"])
            used_mib += int(gpu["used_mib"])
            lines.append(
                f"{state['node_ip'] + ' ' + state['role']:<25} "
                f"{gpu['index']:<3} {gpu['name'][:36]:<36} "
                f"{gpu['temp'] + ' C':>5} {gpu['util'] + '%':>5} "
                f"{gpu['used_mib'] + ' / ' + gpu['total_mib'] + ' MiB':>21} "
                f"{gpu['watts'] + ' W':>8}"
            )
    lines.append("=" * 109)
    lines.append(
        f"Pool: {total_gpus} GPU(s), {used_mib:,} / {total_mib:,} MiB VRAM used"
    )
    for state in states:
        lines.append(
            f"{state['node_ip']}: {state['cpus']} CPUs, "
            f"{state['ram_available_gib']}/{state['ram_total_gib']} GiB RAM available, "
            f"API http://{state['node_ip']}:{api_port}"
        )
    for ip in offline:
        lines.append(f"{ip}: offline")
    return "\n".join(lines)


def status(config: Config) -> None:
    states = []
    offline = []
    for ip in config.peer_ips:
        response = query_node(ip, config.control_port, timeout=3, full=True)
        if response is None:
            offline.append(ip)
        else:
            states.append(response)
    print(format_status(states, offline, config.api_port))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lanllm")
    parser.add_argument("action", choices=("run", "status"))
    args = parser.parse_args(argv)
    try:
        config = Config.from_environment(dict(os.environ))
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.action == "status":
        status(config)
    else:
        signal.signal(signal.SIGUSR1, lambda _signum, _frame: sys.exit(42))
        run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
