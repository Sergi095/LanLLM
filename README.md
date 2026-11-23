# LanLLM

Run a chosen Hugging Face language model across NVIDIA GPUs on computers in
the same LAN. The current setup pools the RTX 5070 Ti and RTX A4000 on
`192.168.2.33` with Katana's RTX 4070 Laptop GPU on `192.168.2.26`.

LanLLM uses Ray to place three pipeline stages and vLLM to serve the model.
Model IDs come directly from Hugging Face; there is no built-in model list.
The model must have an architecture and weight format supported by vLLM, and
its weights plus KV cache must fit the GPUs. The 4070 has 8 GB, so larger
models may need quantization or an uneven layer partition.

## Start the containers

Both machines need Docker, the NVIDIA Container Toolkit, and a CDI GPU
specification. The current hosts are configured; on a new host, generate
`/etc/cdi/nvidia.yaml` with `sudo nvidia-ctk cdi generate
--output=/etc/cdi/nvidia.yaml` and check GPU access with `docker run --rm
--device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi -L`.
The containers use host networking for Ray and NCCL communication. The API
binds to `127.0.0.1:8000` on the head machine by default.

On `192.168.2.33`, in this repository:

```bash
cp .env.head.example .env
docker compose --profile head up -d --build head
```

On Katana, in its copy of this repository:

```bash
cp .env.worker.example .env
docker compose --profile worker up -d --build worker
```

The head container waits until Ray sees all three GPUs, then starts vLLM.
Check startup and cluster placement on the head:

```bash
docker compose logs -f head
docker compose exec head ray status
curl http://127.0.0.1:8000/v1/models
```

Try a chat request on the head:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Use another model

Set `MODEL_ID` to the same Hugging Face repository ID in `.env` on both
machines, then recreate the worker and head containers. Model files are
downloaded to each host's `~/.cache/huggingface` and survive container
restarts. For a gated model, set `HF_TOKEN` in both `.env` files.

Optional settings in `.env` include `HF_REVISION`, `QUANTIZATION`, `DTYPE`,
`MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, and `LAYER_PARTITION`. A partition
must have one hidden-layer count per GPU, totaling the model's layer count;
for example, `10,10,4` for a 24-layer model gives fewer layers to Katana's
8 GB GPU when it is assigned pipeline rank 2. Check the vLLM logs for actual
rank placement before using a custom partition. `TRUST_REMOTE_CODE=1` runs
Python code from a model repository, so set it only for a trusted model.

To expose the API to other LAN clients, set `API_HOST=0.0.0.0` on the head
and restrict access with a firewall or API gateway. Ray's cluster traffic is
intended for a trusted LAN or VPN; host networking is required here. The
head's firewall currently allows incoming TCP from Katana's exact LAN IP.

Stop with `docker compose --profile head down` on the head and
`docker compose --profile worker down` on Katana.

The container image is based on [vLLM's official image](https://docs.vllm.ai/en/latest/deployment/docker/).
Ray placement follows its [on-premise cluster guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).
