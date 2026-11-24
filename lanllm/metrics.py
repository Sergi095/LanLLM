"""Read host CPU and network counters for the cluster dashboard."""

import threading
import time


def read_counters(interface: str) -> tuple[int, int, int, int, float]:
    with open("/proc/stat", encoding="ascii") as source:
        fields = [int(value) for value in source.readline().split()[1:9]]
    total = sum(fields)
    idle = fields[3] + fields[4]
    receive = transmit = 0
    with open("/proc/net/dev", encoding="ascii") as source:
        for line in source:
            name, separator, values = line.partition(":")
            if separator and name.strip() == interface:
                counters = values.split()
                receive, transmit = int(counters[0]), int(counters[8])
                break
    return total, idle, receive, transmit, time.monotonic()


class SystemSampler:
    def __init__(self, interface: str):
        self.interface = interface
        self.previous = read_counters(interface)
        self.lock = threading.Lock()

    def sample(self) -> dict:
        with self.lock:
            current = read_counters(self.interface)
            previous = self.previous
            self.previous = current
        ticks = current[0] - previous[0]
        idle_ticks = current[1] - previous[1]
        seconds = current[4] - previous[4]
        return {
            "cpu_percent": round(100 * (ticks - idle_ticks) / ticks, 1)
            if ticks > 0 else None,
            "network_rx_bps": max(0, round((current[2] - previous[2]) / seconds))
            if seconds > 0 else 0,
            "network_tx_bps": max(0, round((current[3] - previous[3]) / seconds))
            if seconds > 0 else 0,
        }
