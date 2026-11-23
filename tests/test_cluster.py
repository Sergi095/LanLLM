import unittest

from lanllm.cluster import allocate_layers, elect_leader, format_status


class ElectionTests(unittest.TestCase):
    def test_existing_leader_wins_even_when_new_node_has_lower_ip(self) -> None:
        self.assertEqual(
            elect_leader(
                "192.168.2.26", [{"node_ip": "192.168.2.33", "role": "leader"}]
            ),
            "192.168.2.33",
        )

    def test_simultaneous_candidates_choose_same_node(self) -> None:
        self.assertEqual(
            elect_leader(
                "192.168.2.33", [{"node_ip": "192.168.2.26", "role": "candidate"}]
            ),
            "192.168.2.26",
        )
        self.assertEqual(
            elect_leader(
                "192.168.2.26", [{"node_ip": "192.168.2.33", "role": "candidate"}]
            ),
            "192.168.2.26",
        )

    def test_no_peer_promotes_remaining_node(self) -> None:
        self.assertEqual(elect_leader("192.168.2.33", []), "192.168.2.33")

    def test_uneven_vram_gets_uneven_layers_in_either_rank_order(self) -> None:
        self.assertEqual(allocate_layers(48, [8188, 16303, 16376]), [8, 20, 20])
        self.assertEqual(allocate_layers(48, [16303, 16376, 8188]), [20, 20, 8])


class StatusTests(unittest.TestCase):
    def test_status_sums_per_gpu_memory(self) -> None:
        states = [
            {
                "node_ip": "192.168.2.33",
                "role": "leader",
                "model_id": "Qwen/Qwen2.5-14B-Instruct",
                "gpus": [
                    {
                        "index": "0",
                        "name": "RTX 5070 Ti",
                        "total_mib": "16303",
                        "used_mib": "5000",
                        "util": "50",
                        "temp": "55",
                        "watts": "120",
                    }
                ],
                "cpus": 16,
                "ram_total_gib": 32.0,
                "ram_available_gib": 20.0,
            }
        ]
        output = format_status(states, ["192.168.2.26"], 8000)
        self.assertIn("RTX 5070 Ti", output)
        self.assertIn("Model: Qwen/Qwen2.5-14B-Instruct", output)
        self.assertIn("5,000 / 16,303 MiB VRAM used", output)
        self.assertIn("192.168.2.26: offline", output)


if __name__ == "__main__":
    unittest.main()
