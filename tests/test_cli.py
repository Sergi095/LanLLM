import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


class ModelSelectionTests(unittest.TestCase):
    def test_missing_worker_blocks_model_selection(self) -> None:
        command = runpy.run_path(str(Path(__file__).parents[1] / "bin/lanllm"))
        select_model = command["model"]
        leader = {
            "node_ip": "192.168.2.33",
            "role": "leader",
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "gpus": [{"index": "0"}, {"index": "1"}],
        }
        settings = {
            "PEER_IPS": "192.168.2.33,192.168.2.26",
            "TOTAL_GPUS": "3",
        }
        with patch.dict(
            select_model.__globals__,
            {
                "configuration": lambda: settings,
                "query_node": lambda ip, *_args, **_kwargs: (
                    leader if ip == "192.168.2.33" else None
                ),
            },
        ):
            with self.assertRaisesRegex(ValueError, "Only 2/3 GPUs are online"):
                select_model("Qwen/Qwen2.5-14B-Instruct")


if __name__ == "__main__":
    unittest.main()
