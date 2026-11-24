import unittest

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
            "Qwen/Qwen2.5-14B-Instruct",
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


if __name__ == "__main__":
    unittest.main()
