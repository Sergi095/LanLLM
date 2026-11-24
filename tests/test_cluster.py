import json
import http.server
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lanllm.cluster import (
    allocate_layers,
    annotate_model_context,
    ControlServer,
    ProxyServer,
    elect_leader,
    estimate_fit,
    format_status,
    llama_rpc_command,
    llama_server_command,
    model_catalog,
    model_layer_count,
    resolve_partition,
    mount_model_store,
    registered_nodes,
    selected_model,
    selected_model_fit,
    selected_spec,
    verify_model_store,
)


class ElectionTests(unittest.TestCase):
    def test_model_control_requires_worker_only_for_pool_fit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SimpleNamespace(
                node_ip="127.0.0.1", control_port=0,
                peer_ips=["127.0.0.1", "127.0.0.2"],
                local_gpus=2, total_gpus=3,
            )
            state = SimpleNamespace(config=config, identity=lambda: {"role": "leader"})
            server = ControlServer(state)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            models = [
                {
                    "model_id": f"ollama:{name}", "ref": f"ollama:{name}",
                    "fit": fit, "backend": "llama.cpp", "source": "ollama",
                    "relative_path": f"gguf/{name}.gguf", "weight_bytes": 1,
                }
                for name, fit in (("small", "leader"), ("large", "pool"))
            ]
            actual_path = Path
            try:
                with (
                    patch("lanllm.cluster.query_node", return_value=None),
                    patch("lanllm.cluster.registered_nodes", return_value=[]),
                    patch("lanllm.cluster.model_catalog", return_value={"models": models}),
                    patch(
                        "lanllm.cluster.Path",
                        side_effect=lambda value: (
                            root / value.removeprefix("/models/")
                            if value.startswith("/models/")
                            else root if value == "/models" else actual_path(value)
                        ),
                    ),
                ):
                    def select(name: str):
                        body = json.dumps({"model_id": f"ollama:{name}"}).encode()
                        return Request(
                            f"http://127.0.0.1:{server.server_address[1]}/model",
                            data=body,
                        )

                    with urlopen(select("small"), timeout=2) as response:
                        self.assertEqual(response.status, 200)
                    self.assertEqual(
                        json.loads((root / ".lanllm-selection.json").read_text())["ref"],
                        "ollama:small",
                    )
                    with self.assertRaises(HTTPError) as failure:
                        urlopen(select("large"), timeout=2)
                    self.assertEqual(failure.exception.code, 409)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_selected_model_fit_controls_gpu_placement(self) -> None:
        catalog = {"models": [
            {"model_id": "small", "fit": "leader"},
            {"model_id": "large", "fit": "pool"},
        ]}
        self.assertEqual(selected_model_fit(catalog, "small"), "leader")
        self.assertEqual(selected_model_fit(catalog, "large"), "pool")
        self.assertEqual(selected_model_fit(catalog, "missing"), "unknown")

    def test_model_context_is_advertised_for_runtime_discovery(self) -> None:
        listing = {
            "data": [
                {"id": "active", "max_model_len": 16384},
                {"id": "other", "meta": {"n_ctx": 4096}},
            ]
        }
        enriched = annotate_model_context(listing)
        self.assertEqual(enriched["data"][0]["meta"]["n_ctx"], 16384)
        self.assertEqual(enriched["data"][1]["meta"]["n_ctx"], 4096)

    def test_public_proxy_discovers_context_and_forwards_other_paths(self) -> None:
        class Upstream(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = (
                    b'{"data":[{"id":"active","max_model_len":16384}]}'
                    if self.path == "/v1/models" else b"healthy"
                )
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args) -> None:
                pass

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever)
        upstream_thread.start()
        proxy = ProxyServer(
            SimpleNamespace(api_port=0), "127.0.0.1",
            upstream.server_address[1], annotate_models=True,
        )
        proxy_thread = threading.Thread(target=proxy.serve_forever)
        proxy_thread.start()
        try:
            base = f"http://127.0.0.1:{proxy.server_address[1]}"
            with urlopen(base + "/v1/models", timeout=2) as response:
                listing = json.load(response)
            self.assertEqual(listing["data"][0]["meta"]["n_ctx"], 16384)
            with urlopen(base + "/health", timeout=2) as response:
                self.assertEqual(response.read(), b"healthy")
        finally:
            proxy.shutdown()
            proxy.server_close()
            upstream.shutdown()
            upstream.server_close()
            proxy_thread.join(timeout=2)
            upstream_thread.join(timeout=2)

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

    def test_node_with_model_store_becomes_leader(self) -> None:
        self.assertEqual(
            elect_leader(
                "192.168.2.33",
                [
                    {
                        "node_ip": "192.168.2.26",
                        "role": "leader",
                        "store_priority": 0,
                    }
                ],
                local_priority=1,
            ),
            "192.168.2.33",
        )
        self.assertEqual(
            elect_leader(
                "192.168.2.26",
                [
                    {
                        "node_ip": "192.168.2.33",
                        "role": "candidate",
                        "store_priority": 1,
                    }
                ],
            ),
            "192.168.2.33",
        )

    def test_uneven_vram_gets_uneven_layers_in_either_rank_order(self) -> None:
        self.assertEqual(allocate_layers(48, [8188, 16303, 16376]), [8, 20, 20])
        self.assertEqual(allocate_layers(48, [16303, 16376, 8188]), [20, 20, 8])

    def test_multimodal_text_layers_are_used_for_partition(self) -> None:
        self.assertEqual(model_layer_count({"text_config": {"num_hidden_layers": 64}}), 64)

    def test_model_profile_can_override_auto_partition(self) -> None:
        config = SimpleNamespace(
            model_id="owner/repository",
            env={
                "LAYER_PARTITION": "auto",
                "MODEL_PROFILES_JSON": json.dumps({
                    "owner/repository": {"layer_partition": "29,29,6"}
                }),
            },
        )
        self.assertEqual(resolve_partition(config, "192.168.1.10"), "29,29,6")


class StatusTests(unittest.TestCase):
    def test_catalog_lists_downloaded_gguf_model(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "gguf/ollama/library/qwen2.5/0.5b/model.gguf"
            file.parent.mkdir(parents=True)
            file.write_bytes(b"x" * 100)
            (file.parent / "model.gguf.lanllm-ready").write_text(
                json.dumps(
                    {
                        "ref": "ollama:qwen2.5:0.5b",
                        "backend": "llama.cpp",
                        "source": "ollama",
                        "relative_path": "gguf/ollama/library/qwen2.5/0.5b/model.gguf",
                    }
                )
            )
            config = SimpleNamespace(
                node_ip="192.168.2.33", env={"GPU_MEMORY_UTILIZATION": "0.9"}
            )
            nodes = [
                {
                    "node_ip": "192.168.2.33",
                    "online": True,
                    "gpus": [{"total_mib": "16303"}],
                }
            ]
            catalog = model_catalog(config, nodes, root)
        self.assertEqual(catalog["models"][0]["model_id"], "ollama:qwen2.5:0.5b")
        self.assertEqual(catalog["models"][0]["backend"], "llama.cpp")

    def test_llama_commands_use_remote_gpu_and_same_api_port(self) -> None:
        config = SimpleNamespace(
            node_ip="192.168.2.26",
            model_path=Path("/models/gguf/ollama/library/qwen2.5/0.5b/model.gguf"),
            model_id="ollama:qwen2.5:0.5b",
            api_port=8001,
            env={"LLAMA_RPC_PORT": "50052", "MAX_MODEL_LEN": "2048"},
        )
        self.assertEqual(llama_rpc_command(config)[-2:], ["--port", "50052"])
        command = llama_server_command(config, ["192.168.2.26:50052"])
        self.assertIn("192.168.2.26:50052", command)
        self.assertEqual(command[command.index("--port") + 1], "8001")
        self.assertEqual(command[command.index("--alias") + 1], config.model_id)
        self.assertNotIn("--rpc", llama_server_command(config, []))

    def test_json_selection_supersedes_legacy_model(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".lanllm-model").write_text("Qwen/Qwen2.5-14B-Instruct\n")
            selection = {
                "ref": "ollama:qwen2.5:0.5b",
                "backend": "llama.cpp",
                "relative_path": "gguf/ollama/library/qwen2.5/0.5b/model.gguf",
            }
            (root / ".lanllm-selection.json").write_text(json.dumps(selection))
            self.assertEqual(selected_spec(root), selection)

    def test_registry_keeps_last_known_worker_capacity_when_offline(self) -> None:
        config = SimpleNamespace(
            node_ip="192.168.2.33",
            peer_ips=("192.168.2.33", "192.168.2.26"),
            control_port=8501,
        )
        local = {
            "node_ip": "192.168.2.33",
            "role": "leader",
            "gpus": [{"index": "0", "name": "RTX A4000", "total_mib": "16376"}],
            "cpus": 16,
            "ram_total_gib": 32.0,
        }
        worker = {
            "node_ip": "192.168.2.26",
            "role": "worker",
            "gpus": [{"index": "0", "name": "RTX 4070", "total_mib": "8188"}],
            "cpus": 8,
            "ram_total_gib": 16.0,
        }
        state = SimpleNamespace(snapshot=lambda: local)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("lanllm.cluster.query_node", side_effect=[worker, None]):
                first = registered_nodes(config, state, root)
                second = registered_nodes(config, state, root)
        self.assertTrue(first[1]["online"])
        self.assertFalse(second[1]["online"])
        self.assertEqual(second[1]["gpus"], first[1]["gpus"])
        self.assertEqual(second[1]["ram_total_gib"], 16.0)

    def test_weight_fit_distinguishes_leader_and_registered_pool(self) -> None:
        self.assertEqual(estimate_fit(1_000_000_000, 25_314, 30_635), "leader")
        self.assertEqual(estimate_fit(29_540_134_000, 25_314, 30_635), "pool")
        self.assertEqual(estimate_fit(70_000_000_000, 25_314, 30_635), "too_large")

    def test_worker_mounts_leader_advertised_directory(self) -> None:
        config = SimpleNamespace(node_ip="192.168.2.26", control_port=8501)
        leader = {
            "role": "leader",
            "model_store_dir": "/disks/models/lanllm",
        }
        with (
            patch.object(Path, "is_mount", return_value=False),
            patch("lanllm.cluster.query_node", return_value=leader),
            patch("lanllm.cluster.subprocess.run") as run,
            patch("lanllm.cluster.verify_model_store"),
        ):
            mount_model_store(config, "192.168.2.33")
        command = run.call_args.args[0]
        self.assertIn("192.168.2.33:/disks/models/lanllm", command)

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

    def test_unmounted_store_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = Path(directory)
            with self.assertRaisesRegex(ValueError, "LanLLM model store"):
                verify_model_store(store)
            (store / ".lanllm-store").write_text("lanllm-model-store-v1\n")
            verify_model_store(store)

    def test_shared_model_selection_rejects_invalid_ids(self) -> None:
        with TemporaryDirectory() as directory:
            store = Path(directory)
            self.assertIsNone(selected_model(store))
            (store / ".lanllm-model").write_text("Qwen/Qwen2.5-14B-Instruct\n")
            self.assertEqual(selected_model(store), "Qwen/Qwen2.5-14B-Instruct")
            (store / ".lanllm-model").write_text("../other")
            with self.assertRaisesRegex(ValueError, "Invalid model ID"):
                selected_model(store)


if __name__ == "__main__":
    unittest.main()
