# LanLLM

Run Hugging Face and Ollama library models across NVIDIA GPUs on computers in
one LAN. vLLM serves Hugging Face checkpoints; llama.cpp serves GGUF files from
Hugging Face or the Ollama library.

Both hosts run the same Docker image, Compose service, and commands. A node
finds its own LAN address, network interface, and GPU count. If another node
is already leading, it joins that cluster; otherwise the first available
node starts one. The leader runs vLLM or llama.cpp, and other nodes forward API
traffic to it. vLLM uses Ray pipeline stages; llama.cpp uses GPU RPC. The
worker mounts the leader's model folder over NFS using the elected leader IP.

The model ID can be any Hugging Face model with an architecture and weight
format supported by vLLM, provided its weights and KV cache fit the GPUs.
GPU VRAM is distributed across stages, not exposed as one contiguous block.
Models may need quantization or an uneven layer partition when GPUs have
different VRAM capacities. Automatic partitioning assigns layers according to
the available VRAM on each GPU.

## Start on either host

Each host needs Docker, the NVIDIA Container Toolkit, and a CDI GPU
specification. Generate
`/etc/cdi/nvidia.yaml` with `sudo nvidia-ctk cdi generate
--output=/etc/cdi/nvidia.yaml`, then check access with `docker run --rm
--device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi -L`.

`lanllm start` uses passwordless `sudo` to install and start the NFS server if
needed. It uses the local model folder from `lanllm.conf` and exports it
read-only to the other listed nodes. Each container mounts the elected
leader's store at `/models`. The leader reads its local disk; workers read
that same folder over the LAN. The containers need
privileged mode to mount NFS inside their own mount namespaces.

Put this repository on each host. Install the host command once on each:

```bash
cd ~/repos/LanLLM
cp lanllm.conf.example lanllm.conf
sudo ln -s "$PWD/bin/lanllm" /usr/local/bin/lanllm
```

Edit `lanllm.conf` to choose `models.directory`, the default Hugging Face
model, peer IPs, GPU count, inference settings, and API port. The directory
is local to each host; it can differ between hosts. The default is
`/srv/lanllm-store`. LanLLM creates and shares only
its own folder. New downloads
go into its `models/` and `gguf/` subfolders. Keep `.env` for optional secrets
or settings absent from the TOML file; values in `lanllm.conf` take precedence.

Then run these same commands from **any directory** on either host:

```bash
lanllm start
lanllm
```

`lanllm` (or `lanllm status`) prints a combined GPU table directly from the
host, without Docker. `lanllm logs` follows startup logs; `lanllm stop`
stops the local node.

`lanllm nodes` shows how many nodes have registered, how many are online, and
each node's GPU count, total VRAM, CPU count, RAM, and GPU names. The leader
records the last known capacity in its shared model folder, so an offline
worker remains visible as offline. Run the same command on either host.

`lanllm models` lists models downloaded in the leader's model folder and
estimates whether their weight files fit the leader's GPUs alone, need the
registered GPU pool, or exceed that pool. It also marks a pool-only model when
the worker is offline. The estimate applies the configured GPU memory limit
and reserves 2 GiB per GPU for runtime and KV cache; actual inference may
need more memory. It shows the backend for each model. Models that have not
been downloaded are not listed.

`cluster.peers` in `lanllm.conf` lists every node. Set the same model ID and
GPU count on all nodes. The first node can start alone; it waits for the
other GPU inventory and the requested GPU count before serving. Start the
second node whenever ready. If the leader stops, a remaining node can become
leader when the former leader's control endpoint becomes unreachable. The API
is ready again after the requested GPU count rejoins and the model loads.

The table shows the GPU name, temperature, utilization, VRAM usage, power, and
CPU/RAM summary for every reachable node. `nvidia-smi` still shows only the
physical GPUs on the host where it runs.

The OpenAI-compatible API is available on **both** LAN addresses:

```text
http://<node-a-lan-ip>:8001/v1
http://<node-b-lan-ip>:8001/v1
```

Only one inference server can bind port 8001 on a host at a time. Change
`api.port` on every node if another service uses it.

For example, run this on either host:

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Use another model

Download on either host; the leader stores one copy in `models.directory`:

```bash
lanllm download Qwen/Qwen2.5-14B-Instruct
lanllm download ollama:qwen2.5:0.5b
lanllm download https://ollama.com/library/qwen2.5:0.5b
lanllm download owner/gguf-repository --backend llama.cpp --file model.Q4_K_M.gguf
lanllm models
```

Plain Hugging Face IDs and `https://huggingface.co/owner/repository` URLs use
vLLM by default. Meta Llama repositories on Hugging Face work when your account
has access; set `HF_TOKEN` in `lanllm.conf` or `.env` on the leader. For a GGUF
repository, add `--backend llama.cpp`; LanLLM selects its single Q4_K_M file
automatically when unambiguous, otherwise specify `--file`. Ollama library
names and URLs use llama.cpp. The Ollama download extracts the GGUF weight blob
directly into the shared model store without keeping a second Ollama cache.
The current Ollama path supports models with one GGUF weight layer; models that
require extra projector files are rejected.
Run `lanllm download` without a model to follow the current download from
either host.

Downloading does not interrupt the running model. To serve a downloaded model,
copy its reference from `lanllm models` or the download completion message:

```bash
lanllm model switch Qwen/Qwen2.5-14B-Instruct
lanllm model switch ollama:qwen2.5:0.5b
```

`lanllm model switch <ID>` (or the shorter `lanllm switch <ID>`) sends the
selection to the running leader; both nodes restart
their inference processes when it changes. vLLM uses Ray pipeline stages;
llama.cpp exposes each worker's GPUs through its RPC backend on TCP 50052.
Workers read the leader's model folder over NFS without a local disk copy.
For a Hugging Face checkpoint, `lanllm model switch <ID>` can also download the model
automatically when selected. The command requires all GPUs
configured in `cluster.total_gpus` to be online; it rejects a model change if
a worker is absent. Set `models.default_id` before starting the pool to choose
its initial model. Run `lanllm model` without an ID to show the selected model.
`models.default_id` in `lanllm.conf` is the initial default before a selection
is made.
The `[models.profiles."owner/repository"]` tables set model-specific vLLM
options. `tool_call_parser` enables OpenAI `tool_choice: "auto"` requests.
`served_model_names` adds API aliases while retaining the full Hugging Face ID.

To import an existing Hugging Face Hub cache, use the `lanllm.import_cache`
module with the snapshot revision. It
hardlinks the snapshot into LanLLM's shared store, so the files use the same
disk blocks and the original cache remains usable. A separately cached
tokenizer can be imported with `--tokenizer`. The cache and store must be on
the same filesystem.

```bash
cd /path/to/LanLLM
sudo python3 -m lanllm.import_cache owner/repository SNAPSHOT_REVISION \
  --cache /path/to/huggingface/hub --store /path/to/lanllm-store
```

For a gated model that may be served after a leader change, set `HF_TOKEN` on
both hosts.

Optional settings include `HF_REVISION`, `QUANTIZATION`, `DTYPE`,
`MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, and `LAYER_PARTITION`. A partition
has one hidden-layer count per GPU, totaling the model's layer count. Check
the logs for actual pipeline rank placement before assigning an uneven
partition. Set `LAYER_PARTITION=auto` to apportion layers by GPU VRAM with
extra room reserved on small cards. The nodes calculate the same partition
based on the elected leader's GPU order. `TRUST_REMOTE_CODE=1` runs code
from a model repository; set it only for a trusted model.

The API listens on all LAN interfaces. Restrict access with a firewall or
API gateway on networks you do not trust. Ray and the leader election use
host networking. The nodes need TCP access to one another, including NFS on
TCP port 2049 and Ray's dynamic worker ports. If the leader changes to a node
without the selected model, that node downloads it into its own local store.
llama.cpp also needs TCP port 50052 between nodes for RPC. Its RPC backend is
intended for trusted local networks and is still experimental.

Stop with `lanllm stop` on both hosts. To remove LanLLM from a node, run
`lanllm uninstall` there. It stops the container and removes the image,
NFS export, project cache, command link, and repository checkout. For a model
directory created by LanLLM it removes that dedicated directory. Existing
models outside it remain untouched. Run it on every node for a full cluster uninstall. It removes the
NFS server package only when LanLLM installed it and no other exports use it.
Preview the removal with `lanllm uninstall --dry-run`. Use
`lanllm uninstall --keep-source` to preserve the repository checkout.

LanLLM uses [vLLM's official container image](https://docs.vllm.ai/en/latest/deployment/docker/)
and Ray's [on-premise cluster approach](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).
The GGUF backend uses [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md).
