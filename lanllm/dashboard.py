"""Live terminal dashboard for the distributed GPU pool."""

import curses
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from .cluster import query_node


def poll_nodes(peers: list[str], port: int) -> tuple[list[dict], list[str]]:
    with ThreadPoolExecutor(max_workers=max(1, len(peers))) as executor:
        results = list(
            executor.map(
                lambda ip: query_node(ip, port, timeout=1.5, full=True), peers
            )
        )
    return (
        [state for state in results if state is not None],
        [ip for ip, state in zip(peers, results, strict=True) if state is None],
    )


def bar(value: float, width: int = 12) -> str:
    filled = max(0, min(width, round(width * value / 100)))
    return "█" * filled + "░" * (width - filled)


def rate(value: int | float | None) -> str:
    if value is None:
        return "?"
    number = float(value)
    for unit in ("B/s", "KiB/s", "MiB/s", "GiB/s"):
        if number < 1024 or unit == "GiB/s":
            return f"{number:.1f} {unit}"
        number /= 1024
    return "?"


def sparkline(values: deque[float], width: int = 16) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[max(0, min(7, int(value * 7 / 100)))] for value in list(values)[-width:])


def render_dashboard(
    states: list[dict],
    offline: list[str],
    width: int,
    api_port: int,
    history: dict[str, deque[float]],
) -> list[str]:
    online = len(states)
    total = online + len(offline)
    leader = next((state for state in states if state["role"] == "leader"), None)
    model = leader["model_id"] if leader else "waiting for leader"
    lines = [
        f" LanLLM top  |  {online}/{total} nodes online  |  q quit",
        f" Model: {model}",
        "─" * max(1, width),
    ]
    for state in states:
        ip = state["node_ip"]
        cpu = state.get("cpu_percent")
        if cpu is not None:
            history[f"cpu:{ip}"].append(float(cpu))
        cpu_text = f"{cpu:5.1f}%" if cpu is not None else "    ?%"
        ram_total = float(state["ram_total_gib"])
        ram_used = max(0, ram_total - float(state["ram_available_gib"]))
        ram_pct = 100 * ram_used / ram_total if ram_total else 0
        lines.extend(
            [
                f" {ip}  {state['role'].upper()}  |  API :{api_port}",
                f" CPU  {bar(float(cpu or 0))} {cpu_text}    "
                f"RAM {bar(ram_pct)} {ram_used:.1f}/{ram_total:.1f} GiB",
                f" NET  ↓ {rate(state.get('network_rx_bps')):<12} "
                f"↑ {rate(state.get('network_tx_bps')):<12} "
                f"CPU trend {sparkline(history[f'cpu:{ip}'])}",
            ]
        )
        for gpu in state["gpus"]:
            gpu_key = f"gpu:{ip}:{gpu['index']}"
            try:
                util = float(gpu["util"])
                used = int(gpu["used_mib"])
                total_mib = int(gpu["total_mib"])
            except (ValueError, TypeError):
                util, used, total_mib = 0, 0, 0
            history[gpu_key].append(util)
            vram_pct = 100 * used / total_mib if total_mib else 0
            name = gpu["name"]
            lines.extend(
                [
                    f" GPU {gpu['index']}  {name}  {sparkline(history[gpu_key])}",
                    f"      {bar(util)} {util:5.1f}%  "
                    f"VRAM {bar(vram_pct)} {used / 1024:.1f}/{total_mib / 1024:.1f} GiB  "
                    f"{gpu['temp']}°C  {gpu['watts']}W",
                ]
            )
        lines.append("─" * max(1, width))
    for ip in offline:
        lines.append(f" {ip}  OFFLINE")
    if not states:
        lines.append(" No nodes are reachable. Start LanLLM on a configured host.")
    lines.append(" Live CPU, RAM, network and GPU metrics")
    return [line[:width] for line in lines]


def run_dashboard(peers: list[str], port: int, api_port: int, interval: float) -> int:
    history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=30))

    def draw(screen) -> None:
        curses.curs_set(0)
        screen.nodelay(True)
        while True:
            states, offline = poll_nodes(peers, port)
            height, width = screen.getmaxyx()
            lines = render_dashboard(states, offline, max(1, width - 1), api_port, history)
            screen.erase()
            for row, line in enumerate(lines[: max(0, height - 1)]):
                try:
                    screen.addstr(row, 0, line)
                except curses.error:
                    pass
            screen.refresh()
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline:
                key = screen.getch()
                if key in (ord("q"), ord("Q"), 27):
                    return
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    try:
        curses.wrapper(draw)
    except KeyboardInterrupt:
        pass
    return 0
