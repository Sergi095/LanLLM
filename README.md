# LanLLM

Run a Hugging Face language model across NVIDIA GPUs on computers in one LAN.
The current two hosts pool an RTX 5070 Ti and RTX A4000 at `192.168.2.33`
with Katana's RTX 4070 Laptop GPU at `192.168.2.26`.

Both hosts run the same Docker image, Compose service, and commands. A node
finds its own LAN address, network interface, and GPU count. If another node
is already leading, it joins that Ray cluster; otherwise the first available
node starts a cluster. The leader runs vLLM, and other nodes forward API
traffic to it. The model uses pipeline stages across all three GPUs.

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

Put this repository on each host. On **each** host, run the same commands:

```bash
cd ~/repos/LanLLM
cp .env.example .env
docker compose up -d --build
```

`PEER_IPS` in `.env` lists every node. Set the same `MODEL_ID` and
`TOTAL_GPUS` on all nodes. The first node can start alone; it waits for the
other GPU inventory and the requested GPU count before serving. Start the
second node whenever ready. If the leader stops, a remaining node can become
leader when the former leader's control endpoint becomes unreachable. The API
is ready again after the
requested GPU count rejoins and the model loads.

On either host, inspect the pool with a display similar to `nvidia-smi`:

```bash
docker compose exec node python3 -m lanllm status
```

This shows the GPU name, temperature, utilization, VRAM usage, power, and
CPU/RAM summary for every reachable node. `nvidia-smi` still shows only the
physical GPUs on the host where it runs. For startup logs use
`docker compose logs -f node`.

The OpenAI-compatible API is available on **both** LAN addresses:

```text
http://192.168.2.33:8000/v1
http://192.168.2.26:8000/v1
```

For example, run this on either host:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Use another model

Set `MODEL_ID` to the Hugging Face repository ID in `.env` on both hosts,
then run `docker compose up -d --force-recreate` on both hosts. Model files
are cached under each host's `~/.cache/huggingface` and survive restarts.
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
host networking. The nodes need TCP access to one another, including Ray's
dynamic worker ports.

Stop with `docker compose down` on both hosts.

LanLLM uses [vLLM's official container image](https://docs.vllm.ai/en/latest/deployment/docker/)
and Ray's [on-premise cluster approach](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).
