# vLLM pipeline layer partitioning

LanLLM uses vLLM pipeline parallelism when a model requires the GPU pool. The model's hidden layers are divided into consecutive stages, with one stage per GPU. Each stage also needs memory for runtime state; some stages hold embeddings or the output head. A model that fits by weight size alone can still fail during startup or at a larger context size.

This setting applies to vLLM models that use the pool. llama.cpp uses `llama_tensor_split` instead; see [model configuration](model-configuration.md#llamacpp-settings).

## Start with automatic partitioning

Set the default in `lanllm.conf` on each node:

```toml
[inference]
layer_partition = "auto"
```

When the pool is used, LanLLM reads the hidden-layer count from the model's `config.json`, collects GPU memory sizes, and allocates at least one layer to each GPU. Its estimate reserves additional memory on smaller GPUs. Startup logs print the chosen counts as `Auto layer partition: ...`. This is an initial estimate; it does not measure every model-specific allocation.

## Set a manual partition

Use a manual partition if the automatic split starts with an out-of-memory error or leaves a stage with too little cache memory. For example, a model with 48 hidden layers on three pipeline stages could use:

```toml
[models.profiles."owner/repository"]
layer_partition = "20,18,10"
```

The three entries add up to 48. A valid manual split has **one positive integer per GPU** and **the sum equals the model's hidden-layer count**. The values refer to vLLM pipeline rank order. They are not percentages or GPU indices. The hidden-layer count is usually `num_hidden_layers` in the model's `config.json`, or `text_config.num_hidden_layers` for some multimodal models. vLLM checks the entry count and sum during startup.

To tune the split:

1. Start with `auto` and record the printed partition and any GPU memory error from `lanllm logs`.
2. Determine which pipeline rank ran out of memory and identify its GPU from the startup logs. Check the rank mapping after a leadership or device-order change.
3. Move one or two layers from the constrained rank to a rank with more free memory. Keep the total unchanged.
4. Put the same model profile on every node. Run `lanllm start` on each node to apply the updated configuration. Select the model with `lanllm switch owner/repository` if it is not already active, then check `lanllm status` and a short inference request.
5. Recheck the split after changing the model revision, quantization, context size, GPU inventory, or leader. A manual split is specific to that model and device order.

For example, if the third stage cannot load with `20,18,10` and the first stage has spare memory, try `21,18,9`. The right answer depends on the model's non-layer weights and runtime memory; layer counts alone cannot prove that a split will fit.

## Related settings

`max_model_len` determines the maximum context size and strongly affects KV cache memory. `gpu_memory_utilization` limits the fraction of each GPU's memory vLLM may use. `max_num_seqs` and `max_num_batched_tokens` can reduce peak scheduling memory. Tune these before assuming that every startup failure is caused by layer placement.

For a model that fits on the leader's GPUs, LanLLM starts it there without waiting for pool workers; a pool-specific layer partition is then unnecessary. The `lanllm models` fit result is an estimate, so confirm placement and readiness from the command output and logs.

Further reading: [vLLM parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) and [vLLM's pipeline partition implementation](https://docs.vllm.ai/en/latest/api/vllm/distributed/index.html).
