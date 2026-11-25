# Model configuration

LanLLM serves Hugging Face checkpoints through vLLM and single-file GGUF models through llama.cpp. It can download GGUF files from Hugging Face repositories and compatible entries in the Ollama library. Model source, backend, and runtime settings are separate choices: selecting a source does not automatically determine a safe context size or memory configuration.

## Configure and serve a model

Run these commands on a node with the `lanllm` command installed:

```bash
lanllm fit owner/repository
lanllm download owner/repository
lanllm models
lanllm switch owner/repository
```

For a GGUF file in a Hugging Face repository, specify the backend and filename:

```bash
lanllm fit owner/gguf-repository --backend llama.cpp --file model.Q4_K_M.gguf
lanllm download owner/gguf-repository --backend llama.cpp --file model.Q4_K_M.gguf
lanllm models
lanllm switch 'hf:owner/gguf-repository@model.Q4_K_M.gguf'
```

For an Ollama library entry:

```bash
lanllm fit ollama:model:tag
lanllm download ollama:model:tag
lanllm switch ollama:model:tag
```

`lanllm fit` estimates whether the weights fit on the leader or require the pool. It is a screening step, not a guarantee: runtime allocations, context cache, and model architecture also affect whether the model starts. `lanllm switch` waits for the model API to become ready. Run `lanllm logs` on a node to inspect startup errors; press Ctrl-C to stop following logs.

## Configuration files and precedence

Edit `lanllm.conf` in the repository checkout on each node. The installer creates it from `lanllm.conf.example` when it is missing. Settings in `lanllm.conf` take precedence over the corresponding values in `.env`. Node-local paths may differ; the cluster peer list and GPU count should agree across nodes. Keep model profiles consistent across nodes, especially when using the GPU pool.

After editing the configuration, run `lanllm start` on each node. This recreates containers as needed so they receive the new settings. Switching to a model that is already active does not reload an unchanged container's configuration.

```toml
[models]
directory = "/srv/lanllm-store"
default_id = "owner/repository"

[inference]
max_model_len = 8192
gpu_memory_utilization = 0.9
layer_partition = "auto"

[models.profiles."owner/repository"]
max_model_len = 4096
served_model_names = ["friendly-name"]
```

`models.directory` is a local absolute path without spaces. LanLLM creates the model library there and exports it when that node leads; workers mount the leader's library. A model profile is keyed by the exact downloaded Hugging Face model ID. `models.default_id` is the initial selection before a selection is saved. Changing it does not switch an already selected model; use `lanllm switch` for that.

## vLLM model profiles

Profiles under `[models.profiles."owner/repository"]` apply to vLLM models. A profile value overrides the corresponding global inference value where supported. Omit settings that do not apply to a model. Check the model's documentation before setting a tool or reasoning parser.

| Setting | Purpose |
| --- | --- |
| `max_model_len` | Maximum total tokens per request, including prompt and generated output. Larger values require more KV cache memory. |
| `layer_partition` | Comma-separated hidden-layer counts for pooled vLLM pipeline stages, or `"auto"`. See [pipeline partitioning](pipeline-partitioning.md). |
| `served_model_names` | Additional API model names. The model ID is included automatically. |
| `tool_call_parser` | Parser for the model's tool-call format. When set, LanLLM also enables vLLM automatic tool choice. |
| `reasoning_parser` | Parser for the model's reasoning output, if supported. |
| `tokenizer` | Alternate tokenizer path or ID when the model requires one. |
| `dtype`, `kv_cache_dtype` | Weight computation and KV cache data types supported by vLLM and the model. |
| `max_num_seqs`, `max_num_batched_tokens` | Limits for concurrent sequences and batched tokens. Reducing them can lower peak memory use. |
| `cpu_offload_gb` | vLLM CPU weight offload allowance; may reduce GPU memory pressure at a speed cost. |
| `gdn_prefill_backend` | Advanced, model-specific vLLM backend selection. Set only when a compatible backend is known. |
| `trust_remote_code` | Allow custom repository code. Use only for a trusted source. |

Example profile for a model whose documented tool-call format matches a supported parser:

```toml
[models.profiles."owner/repository"]
max_model_len = 8192
tool_call_parser = "hermes"
max_num_seqs = 1
```

The example parser is illustrative; another model may need a different parser or none. Global `[inference]` values provide defaults for `max_model_len`, `layer_partition`, `dtype`, and `trust_remote_code`. Set `gpu_memory_utilization` and, if needed, `quantization` in `[inference]` for vLLM.

## llama.cpp settings

GGUF and Ollama models use llama.cpp. The vLLM model profiles above do not configure this backend. Set these options in `[inference]` on the node that may lead:

```toml
[inference]
max_model_len = 8192
llama_parallel = 1
# llama_tensor_split = "2,2,1"
# llama_cache_type_k = "q8_0"
# llama_cache_type_v = "q8_0"
# llama_batch_size = 512
# llama_ubatch_size = 128
```

`max_model_len` sets the llama.cpp context size. `llama_parallel` sets concurrent request slots; increasing it uses more context memory. `llama_tensor_split` supplies relative GPU weight proportions in llama.cpp device order, not vLLM layer counts. Leave it unset until placement needs tuning, then check startup logs to confirm device order. Smaller batch and microbatch values can reduce prompt-processing memory peaks. KV cache types can reduce context memory if the model and backend support them.

## Check the active service

```bash
lanllm models
lanllm status
curl http://<node-lan-ip>:8001/v1/models
```

The model list shows downloaded weights and fit estimates. The API response lists the active model names and context limit. Configure each client to use an active model name and stay within its served context limit; system messages and tool schemas use tokens too.

If a model does not start, use `lanllm logs` to find the first error. For GPU memory errors, try a smaller context, lower batch or concurrency limits, a smaller model variant, or a revised [pipeline partition](pipeline-partitioning.md). For automatic tool-choice errors, check that the model profile has a compatible `tool_call_parser`.
