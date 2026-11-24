"""Discover the LAN pool, elect a Ray head, and expose one API on every node."""

import argparse
import csv
import fcntl
import http.server
import json
import os
import re
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


def selected_model(root: Path = Path("/models")) -> str | None:
    selection = root / ".lanllm-model"
    try:
        model_id = selection.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not MODEL_ID_PATTERN.fullmatch(model_id) or ".." in model_id:
        raise ValueError(f"Invalid model ID in {selection}: {model_id!r}")
    return model_id


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

    @property
    def model_path(self) -> Path:
        return Path("/models/models") / self.model_id

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
            local_store / "models" / self.config.model_id / ".lanllm-ready"
        ).is_file()
        has_selection = (local_store / ".lanllm-model").is_file()
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
            "gpus": gpu_rows(),
            "cpus": os.cpu_count(),
            "ram_total_gib": round(total_ram, 1),
            "ram_available_gib": round(available_ram, 1),
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
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path != "/model":
            self.send_error(404)
            return
        if self.client_address[0] not in self.server.state.config.peer_ips:
            self.send_error(403)
            return
        if self.server.state.identity()["role"] != "leader":
            self.send_error(409, "This node is not the leader")
            return
        config = self.server.state.config
        online_gpus = config.local_gpus
        for ip in config.peer_ips:
            if ip == config.node_ip:
                continue
            peer = query_node(ip, config.control_port, full=True)
            if peer is not None:
                online_gpus += len(peer["gpus"])
        if online_gpus < config.total_gpus:
            self.send_error(
                409,
                f"Only {online_gpus}/{config.total_gpus} GPUs are online",
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 512:
                raise ValueError("Invalid model request length")
            model_id = json.loads(self.rfile.read(length))["model_id"]
            if (
                not isinstance(model_id, str)
                or not MODEL_ID_PATTERN.fullmatch(model_id)
                or ".." in model_id
            ):
                raise ValueError("Invalid Hugging Face model ID")
        except (ValueError, KeyError, TypeError) as exc:
            self.send_error(400, str(exc))
            return
        temporary = Path("/models") / (
            f".lanllm-model.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            temporary.write_text(model_id + "\n", encoding="utf-8")
            temporary.replace(Path("/models/.lanllm-model"))
        except OSError as exc:
            self.send_error(503, str(exc))
            return
        payload = json.dumps({"model_id": model_id}).encode()
        self.send_response(200)
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
    requested = config.env.get("LAYER_PARTITION", "").strip()
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
    layer_count = model_config.get("num_hidden_layers")
    if not isinstance(layer_count, int):
        raise ValueError(
            "Auto layer partition requires num_hidden_layers in config.json"
        )
    partition = ",".join(str(count) for count in allocate_layers(layer_count, memory))
    print(f"Auto layer partition: {partition} for GPUs {memory} MiB", flush=True)
    return partition


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


def model_catalog(config: Config, nodes: list[dict]) -> dict:
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
    root = Path("/models/models")
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
                if file.is_file() and file.suffix in {".safetensors", ".bin", ".gguf"}
            ]
            weight_bytes = sum(file.stat().st_size for file in weights)
            weight_mib = (weight_bytes + 1048575) // 1048576
            models.append(
                {
                    "model_id": directory.relative_to(root).as_posix(),
                    "weight_bytes": weight_bytes,
                    "fit": estimate_fit(weight_bytes, leader_mib, pool_mib),
                    "fits_online_vram": bool(weights) and weight_mib <= online_mib,
                }
            )
    return {
        "models": models,
        "leader_usable_mib": leader_mib,
        "registered_usable_mib": pool_mib,
        "online_usable_mib": online_mib,
        "registered_nodes": sum(bool(node["gpus"]) for node in nodes),
        "online_nodes": sum(node["online"] for node in nodes),
    }


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
    revision = config.env.get("HF_REVISION", "").strip() or "main"
    try:
        return (config.model_path / "config.json").is_file() and (
            config.model_path / ".lanllm-ready"
        ).read_text().strip() == revision
    except OSError:
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
        try:
            upstream = socket.create_connection(
                (self.server.leader_ip, self.server.api_port), timeout=3
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


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, config: Config, leader_ip: str):
        self.leader_ip = leader_ip
        self.api_port = config.api_port
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


def lower_priority_leader(config: Config) -> str | None:
    local_store = Path("/models-local")
    local_priority = int(
        (local_store / ".lanllm-model").is_file()
        or (local_store / "models" / config.model_id / ".lanllm-ready").is_file()
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
        if available_gpus >= config.total_gpus:
            break
        print(
            f"Waiting for GPU inventory: {available_gpus}/{config.total_gpus}",
            flush=True,
        )
        time.sleep(5)
    registered_nodes(config, state)
    ensure_model(config)
    settings = config.settings(
        "head", config.node_ip, resolve_partition(config, config.node_ip)
    )
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
                vllm_command(settings), env=runtime_environment(settings)
            )
            last_registry_update = 0.0
            while server.poll() is None:
                other_leader = lower_priority_leader(config)
                if other_leader is not None:
                    print(f"Another leader won election at {other_leader}", flush=True)
                    return
                if time.monotonic() - last_registry_update >= 15:
                    registered_nodes(config, state)
                    last_registry_update = time.monotonic()
                time.sleep(3)
            print(f"vLLM exited with code {server.returncode}; restarting", flush=True)
    finally:
        stop_process(server)
        stop_ray()


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
                changed = selected_model() not in (None, state.config.model_id)
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
                active = replace(config, model_id=selected_model() or config.model_id)
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
        leaders = [state["node_ip"] for state in states if state["role"] == "leader"]
        leader = leaders[0] if leaders else "starting"
        lines.append(f"Model: {states[0]['model_id']}   Leader: {leader}")
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
