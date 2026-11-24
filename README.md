# LanLLM

Run a Hugging Face language model across NVIDIA GPUs on computers in one LAN.
The current two hosts pool an RTX 5070 Ti and RTX A4000 at `192.168.2.33`
with Katana's RTX 4070 Laptop GPU at `192.168.2.26`.

Both hosts run the same Docker image, Compose service, and commands. A node
finds its own LAN address, network interface, and GPU count. If another node
is already leading, it joins that Ray cluster; otherwise the first available
node starts a cluster. The leader runs vLLM, and other nodes forward API
traffic to it. The model uses pipeline stages across all three GPUs. The
worker mounts the leader's model folder over NFS using the elected leader IP.

The model ID can be any Hugging Face model with an architecture and weight
format supported by vLLM, provided its weights and KV cache fit the GPUs.
GPU VRAM is distributed across stages, not exposed as one contiguous block.
Katana has 8 GB, so larger models may need quantization or an uneven layer
partition. The current default, [Qwen2.5 14B Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), has about
30 GB of weights and 48 transformer layers. Auto partitioning assigns 8
layers to the 8 GB GPU and 20 to each 16 GB GPU, in whichever pipeline order
the leader creates.

## Start on either host

Both hosts need Docker, the NVIDIA Container Toolkit, and a CDI GPU
specification. The current hosts are configured. On a new host, generate
`/etc/cdi/nvidia.yaml` with `sudo nvidia-ctk cdi generate
--output=/etc/cdi/nvidia.yaml`, then check access with `docker run --rm
--device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi -L`.

`lanllm start` uses passwordless `sudo` to install and start the NFS server if
needed. It creates the local model folder from `lanllm.conf` and exports it
read-only to the other listed nodes. Each container mounts the elected
leader's store at `/models`. The leader reads its local disk; workers read
that same folder over the LAN. No separate NAS is needed. The containers need
privileged mode to mount NFS inside their own mount namespaces.

Put this repository on each host. Install the host command once on each:

```bash
cd ~/repos/LanLLM
cp lanllm.conf.example lanllm.conf
sudo ln -s "$PWD/bin/lanllm" /usr/local/bin/lanllm
```

Edit `lanllm.conf` to choose `models.directory`, the default Hugging Face
model, peer IPs, GPU count, inference settings, and API port. The directory
is local to each host; it can differ between hosts. Set it before downloading
models. Keep `.env` for optional secrets or settings absent from the TOML
file; values in `lanllm.conf` take precedence.

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
need more memory. Models that have not been downloaded are not listed.

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
http://192.168.2.33:8001/v1
http://192.168.2.26:8001/v1
```

Port 8001 matches the inference containers in `repos/vllm`. Its Ollama
embedding service uses 11433 separately. Only one inference server can bind
port 8001 on a host at a time.

For example, run this on either host:

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Use another model

Run `lanllm model <Hugging-Face-repository-ID>` on either host. For example:

```bash
lanllm model Qwen/Qwen2.5-14B-Instruct
```

A `https://huggingface.co/<owner>/<model>` URL works too.

The command sends the selection to the running leader. Running nodes restart
themselves when it changes. The leader downloads a missing model once to
`<models.directory>/models/<MODEL_ID>`; workers read the weights through the
leader's NFS export without a local disk copy. The command requires all GPUs
configured in `cluster.total_gpus` to be online; it rejects a model change if
a worker is absent. Set `models.default_id` before starting the pool to choose
its initial model. Run `lanllm model` without an ID to show the selected model.
`models.default_id` in `lanllm.conf` is the initial default before a selection
is made.
For a gated model, set `HF_TOKEN` in both `.env` files.

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

Stop with `lanllm stop` on both hosts. To remove LanLLM from a node, run
`lanllm uninstall` there. It stops the container and removes the image,
NFS export, local model folder, project cache, command link, and repository
checkout. Run it on every node for a full cluster uninstall. It removes the
NFS server package only when LanLLM installed it and no other exports use it.
Preview the removal with `lanllm uninstall --dry-run`. Use
`lanllm uninstall --keep-source` to preserve the repository checkout.

LanLLM uses [vLLM's official container image](https://docs.vllm.ai/en/latest/deployment/docker/)
and Ray's [on-premise cluster approach](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).
