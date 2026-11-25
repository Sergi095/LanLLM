import unittest
import json

from lanllm.launch import Settings, ray_command, runtime_environment, vllm_command


class LaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings.from_environment(
            "head",
            {
                "NODE_IP": "192.168.2.33",
                "HEAD_IP": "192.168.2.33",
                "NETWORK_INTERFACE": "enp12s0",
                "LOCAL_GPUS": "2",
                "TOTAL_GPUS": "3",
                "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
            },
        )

    def test_three_gpu_model_command(self) -> None:
        command = vllm_command(self.settings)
        self.assertEqual(command[command.index("--pipeline-parallel-size") + 1], "3")
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(
            command[command.index("--distributed-executor-backend") + 1], "ray"
        )
        self.assertIn("--enforce-eager", command)

    def test_nas_model_keeps_hugging_face_api_name(self) -> None:
        settings = Settings.from_environment(
            "head",
            {
                "NODE_IP": "192.168.2.33",
                "HEAD_IP": "192.168.2.33",
                "NETWORK_INTERFACE": "enp12s0",
                "LOCAL_GPUS": "2",
                "TOTAL_GPUS": "3",
                "MODEL_ID": "Qwen/Qwen2.5-14B-Instruct",
                "MODEL_SOURCE": "/models/models/Qwen/Qwen2.5-14B-Instruct",
            },
        )
        command = vllm_command(settings)
        self.assertEqual(command[2], "/models/models/Qwen/Qwen2.5-14B-Instruct")
        self.assertEqual(
            command[command.index("--served-model-name") + 1],
            "lanllm",
        )
        self.assertIn("Qwen/Qwen2.5-14B-Instruct", command)

    def test_model_profile_enables_goose_tools_and_existing_alias(self) -> None:
        settings = Settings.from_environment(
            "head",
            {
                "NODE_IP": "192.168.2.33",
                "HEAD_IP": "192.168.2.33",
                "NETWORK_INTERFACE": "enp12s0",
                "LOCAL_GPUS": "2",
                "TOTAL_GPUS": "3",
                "MODEL_ID": "Qwen/Qwen3-8B",
                "MODEL_SOURCE": "/models/models/Qwen/Qwen3-8B",
                "MODEL_PROFILES_JSON": json.dumps({
                    "Qwen/Qwen3-8B": {
                        "served_model_names": ["qwen3.8"],
                        "tool_call_parser": "hermes",
                        "reasoning_parser": "qwen3",
                        "kv_cache_dtype": "fp8",
                        "max_model_len": 8192,
                        "gdn_prefill_backend": "triton",
                    }
                }),
            },
        )
        command = vllm_command(settings)
        self.assertEqual(command[command.index("--served-model-name") + 1:][:3], ["qwen3.8", "lanllm", "Qwen/Qwen3-8B"])
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "hermes")
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "qwen3")
        self.assertEqual(command[command.index("--max-model-len") + 1], "8192")
        self.assertEqual(
            json.loads(command[command.index("--additional-config") + 1]),
            {"gdn_prefill_backend": "triton"},
        )

    def test_worker_joins_head_with_one_gpu(self) -> None:
        settings = Settings.from_environment(
            "worker",
            {
                "NODE_IP": "192.168.2.26",
                "HEAD_IP": "192.168.2.33",
                "NETWORK_INTERFACE": "enp3s0",
                "LOCAL_GPUS": "1",
                "TOTAL_GPUS": "3",
                "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
            },
        )
        command = ray_command(settings)
        self.assertIn("--address=192.168.2.33:6379", command)
        self.assertIn("--num-gpus=1", command)
        self.assertIn("--block", command)
        self.assertEqual(runtime_environment(settings)["VLLM_HOST_IP"], "192.168.2.26")

    def test_invalid_partition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one positive count"):
            Settings.from_environment(
                "head",
                {
                    "NODE_IP": "192.168.2.33",
                    "HEAD_IP": "192.168.2.33",
                    "NETWORK_INTERFACE": "enp12s0",
                    "LOCAL_GPUS": "2",
                    "TOTAL_GPUS": "3",
                    "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
                    "LAYER_PARTITION": "12,12",
                },
            )

    def test_model_profile_overrides_layer_partition(self) -> None:
        settings = Settings.from_environment(
            "head",
            {
                "NODE_IP": "192.168.2.33",
                "HEAD_IP": "192.168.2.33",
                "NETWORK_INTERFACE": "enp12s0",
                "LOCAL_GPUS": "2",
                "TOTAL_GPUS": "3",
                "MODEL_ID": "Qwen/Qwen3.8-27B-FP8",
                "LAYER_PARTITION": "auto",
                "MODEL_PROFILES_JSON": json.dumps({
                    "Qwen/Qwen3.8-27B-FP8": {
                        "layer_partition": "29,31,4",
                        "linear_backend": "marlin",
                        "language_model_only": True,
                        "gpu_memory_utilization": 0.94,
                    }
                }),
            },
        )
        self.assertEqual(settings.layer_partition, "29,31,4")
        self.assertEqual(runtime_environment(settings)["VLLM_PP_LAYER_PARTITION"], "29,31,4")
        self.assertEqual(
            json.loads(vllm_command(settings)[vllm_command(settings).index("--kernel-config") + 1]),
            {"linear_backend": "marlin"},
        )
        self.assertIn("--language-model-only", vllm_command(settings))
        self.assertEqual(
            vllm_command(settings)[vllm_command(settings).index("--gpu-memory-utilization") + 1],
            "0.94",
        )


if __name__ == "__main__":
    unittest.main()
