"""Live terminal dashboard for the distributed GPU pool."""

import curses
import textwrap
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


def _number(value: object, default: float = 0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _middle(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width < 5:
        return text[:width]
    left = (width - 1) // 2
    return text[:left] + "…" + text[-(width - left - 1):]


def render_dashboard(
    states: list[dict],
    offline: list[str],
    width: int,
    api_port: int,
    history: dict[str, deque[float]],
) -> list[str]:
    width = max(20, width)
    inside = width - 4
    lines: list[str] = []

    def rule(label: str, first: str = "├", last: str = "┤") -> None:
        heading = f"─ {label} " if label else "─"
        lines.append(first + heading[:width - 2] + "─" * max(0, width - 2 - len(heading)) + last)

    def row(content: str = "") -> None:
        lines.append("│ " + content[:inside].ljust(inside) + " │")

    def wrapped(label: str, value: str) -> None:
        prefix = f"{label:<6} "
        chunks = textwrap.wrap(value, max(1, inside - len(prefix)), break_long_words=True)
        for index, chunk in enumerate(chunks or [""]):
            row((prefix if index == 0 else " " * len(prefix)) + chunk)

    online = len(states)
    total = online + len(offline)
    leader = next((state for state in states if state["role"] == "leader"), None)
    model = leader["model_id"] if leader else "waiting for leader"
    all_gpus = [gpu for state in states for gpu in state.get("gpus", [])]
    used_mib = sum(_number(gpu.get("used_mib")) for gpu in all_gpus)
    total_mib = sum(_number(gpu.get("total_mib")) for gpu in all_gpus)
    rule(f"LanLLM   {online}/{total} nodes online", "╭", "╮")
    source, separator, filename = model.partition("@")
    wrapped("MODEL", source)
    if separator:
        wrapped("FILE", filename)
    if leader:
        row(f"BACKEND  {leader.get('backend', '?')}    API  :{api_port}")
    rule(f"POOL   {len(all_gpus)} GPUs")
    pool_pct = 100 * used_mib / total_mib if total_mib else 0
    row(
        f"VRAM  {bar(pool_pct, 20)}  "
        f"{used_mib / 1024:.1f} / {total_mib / 1024:.1f} GiB"
    )
    for state in states:
        ip = state["node_ip"]
        cpu = state.get("cpu_percent")
        if cpu is not None:
            history[f"cpu:{ip}"].append(float(cpu))
        cpu_text = f"{_number(cpu):.0f}%" if cpu is not None else "?%"
        ram_total = _number(state.get("ram_total_gib"))
        ram_used = max(0, ram_total - _number(state.get("ram_available_gib")))
        ram_pct = 100 * ram_used / ram_total if ram_total else 0
        rule(f"{ip}   {state['role'].upper()}")
        cpu_part = f"CPU   {bar(_number(cpu), 10)} {cpu_text:>4}"
        ram_part = f"RAM   {bar(ram_pct, 10)} {ram_used:.1f}/{ram_total:.1f} GiB"
        if inside >= 65:
            row(f"{cpu_part}    {ram_part}")
        else:
            row(cpu_part)
            row(ram_part)
        trend = history[f"cpu:{ip}"]
        trend_text = f"    CPU {sparkline(trend, 10)}" if len(trend) >= 4 else ""
        row(
            f"NET   ↓ {rate(state.get('network_rx_bps')):<12} "
            f"↑ {rate(state.get('network_tx_bps')):<12}{trend_text}"
        )
        for gpu in state.get("gpus", []):
            gpu_key = f"gpu:{ip}:{gpu['index']}"
            util = _number(gpu.get("util"))
            used = _number(gpu.get("used_mib"))
            gpu_total_mib = _number(gpu.get("total_mib"))
            history[gpu_key].append(util)
            vram_pct = 100 * used / gpu_total_mib if gpu_total_mib else 0
            temp = f"{gpu.get('temp', '?')}°C"
            power = f"{_number(gpu.get('watts')):.0f}W"
            prefix = f"GPU {gpu['index']}  "
            name_width = max(5, inside - len(prefix) - 2 - 5 - 2 - 4)
            row(
                f"{prefix}{_middle(str(gpu.get('name', '?')), name_width):<{name_width}}"
                f"  {temp:>5}  {power:>4}"
            )
            util_part = f"       UTIL {bar(util, 10)} {util:3.0f}%"
            vram_part = f"VRAM {bar(vram_pct, 10)} {used / 1024:.1f}/{gpu_total_mib / 1024:.1f} GiB"
            if inside >= 65:
                row(f"{util_part}    {vram_part}")
            else:
                row(util_part)
                row(f"       {vram_part}")
    for ip in offline:
        rule(f"{ip}   OFFLINE")
        row("No metrics available from this node")
    if not states:
        row("No nodes are reachable. Start LanLLM on a configured host.")
    rule("q quit   •   live metrics", "╰", "╯")
    return [line[:width] for line in lines]


def run_dashboard(peers: list[str], port: int, api_port: int, interval: float) -> int:
    history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=30))

    def draw(screen) -> None:
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = curses.COLOR_BLACK
            curses.init_pair(1, curses.COLOR_CYAN, background)
            curses.init_pair(2, curses.COLOR_GREEN, background)
            curses.init_pair(3, curses.COLOR_YELLOW, background)
        screen.nodelay(True)
        while True:
            states, offline = poll_nodes(peers, port)
            height, width = screen.getmaxyx()
            lines = render_dashboard(states, offline, max(1, width - 1), api_port, history)
            screen.erase()
            for row, line in enumerate(lines[:height]):
                try:
                    style = 0
                    if curses.has_colors():
                        if "OFFLINE" in line:
                            style = curses.color_pair(3) | curses.A_BOLD
                        elif "LEADER" in line:
                            style = curses.color_pair(2) | curses.A_BOLD
                        elif line.startswith(("╭", "├", "╰")):
                            style = curses.color_pair(1) | curses.A_BOLD
                    screen.addstr(row, 0, line, style)
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
