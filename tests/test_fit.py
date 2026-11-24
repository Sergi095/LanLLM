import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lanllm.fit import choose_fit
from lanllm.metrics import SystemSampler


class FitTests(unittest.TestCase):
    def test_recommends_largest_gguf_with_headroom(self) -> None:
        gib = 2**30
        siblings = [
            SimpleNamespace(rfilename="model.Q8_K_P.gguf", size=29 * gib),
            SimpleNamespace(rfilename="model.Q6_K_P.gguf", size=24 * gib),
            SimpleNamespace(rfilename="mmproj-model.gguf", size=gib),
        ]
        hub = SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(siblings=siblings)
        )
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            plan = choose_fit("owner/model", None, "", 25 * gib, 30 * gib)
            explicit = choose_fit(
                "owner/model", "llama.cpp", "model.Q8_K_P.gguf", 25 * gib, 30 * gib
            )
        self.assertEqual(plan["recommended"]["file"], "model.Q6_K_P.gguf")
        self.assertEqual(len(plan["variants"]), 2)
        self.assertEqual(explicit["recommended"]["file"], "model.Q8_K_P.gguf")
        self.assertTrue(explicit["recommended"]["tight"])

    def test_full_precision_and_ollama_sources(self) -> None:
        hub = SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(
                siblings=[SimpleNamespace(rfilename="model.safetensors", size=500)]
            )
        )
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            plan = choose_fit("owner/model", None, "", 700, 1000)
        self.assertEqual(plan["recommended"]["backend"], "vllm")
        with patch("lanllm.fit.ollama_manifest", return_value={"size": 400}):
            plan = choose_fit("ollama:example:latest", None, "", 500, 1000)
        self.assertEqual(plan["recommended"]["backend"], "llama.cpp")


class MetricsTests(unittest.TestCase):
    def test_cpu_and_network_rates_use_counter_deltas(self) -> None:
        with patch(
            "lanllm.metrics.read_counters",
            side_effect=[(100, 40, 1000, 2000, 1.0), (200, 60, 3000, 4000, 3.0)],
        ):
            sampler = SystemSampler("eth0")
            sample = sampler.sample()
        self.assertEqual(sample["cpu_percent"], 80.0)
        self.assertEqual(sample["network_rx_bps"], 1000)
        self.assertEqual(sample["network_tx_bps"], 1000)


if __name__ == "__main__":
    unittest.main()
