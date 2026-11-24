import runpy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch


class ModelSelectionTests(unittest.TestCase):
    def test_switch_waits_until_both_apis_advertise_selected_model(self) -> None:
        command = runpy.run_path(str(Path(__file__).parents[1] / "bin/lanllm"))
        wait = command["wait_for_model_ready"]
        old = json.dumps({"data": [{"id": "old"}]}).encode()
        selected = json.dumps({"data": [{"id": "new"}]}).encode()
        responses = iter([
            io.BytesIO(old), io.BytesIO(selected),
            io.BytesIO(selected), io.BytesIO(selected),
        ])
        with patch.dict(
            wait.__globals__,
            {"urlopen": lambda *_args, **_kwargs: next(responses)},
        ), patch.object(wait.__globals__["time"], "sleep") as sleep:
            wait({"PEER_IPS": "192.168.2.33,192.168.2.26"}, "new")
        sleep.assert_called_once_with(2)

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
                "leader_endpoint": lambda *_args: {
                    "models": [{"model_id": "Qwen/Qwen2.5-14B-Instruct", "fit": "pool"}]
                },
                "query_node": lambda ip, *_args, **_kwargs: (
                    leader if ip == "192.168.2.33" else None
                ),
            },
        ):
            with self.assertRaisesRegex(ValueError, "Only 2/3 GPUs are online"):
                select_model("Qwen/Qwen2.5-14B-Instruct")

    def test_leader_fit_model_can_switch_without_worker(self) -> None:
        command = runpy.run_path(str(Path(__file__).parents[1] / "bin/lanllm"))
        select_model = command["model"]
        leader = {
            "node_ip": "192.168.2.33",
            "role": "leader",
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "gpus": [{"index": "0"}, {"index": "1"}],
        }
        settings = {"PEER_IPS": "192.168.2.33,192.168.2.26", "TOTAL_GPUS": "3"}
        with patch.dict(
            select_model.__globals__,
            {
                "configuration": lambda: settings,
                "leader_endpoint": lambda *_args: {
                    "models": [{"model_id": "Qwen/Qwen3-8B", "fit": "leader"}]
                },
                "query_node": lambda ip, *_args, **_kwargs: (
                    leader if ip == "192.168.2.33" else None
                ),
                "post_leader": lambda *_args: {},
                "wait_for_model_ready": lambda *_args, **kwargs: self.assertEqual(
                    kwargs["peers"], ["192.168.2.33"]
                ),
            },
        ):
            self.assertEqual(select_model("Qwen/Qwen3-8B"), 0)


if __name__ == "__main__":
    unittest.main()
