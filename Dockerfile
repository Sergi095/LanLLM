FROM vllm/vllm-openai:v0.33.0

RUN uv pip install --system 'ray==2.60.0'

WORKDIR /opt/lanllm
COPY lanllm ./lanllm

ENTRYPOINT ["python3", "-m", "lanllm"]
