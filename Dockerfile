FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS llama-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp/llama.cpp
RUN git init && git fetch --depth 1 https://github.com/ggml-org/llama.cpp.git \
    2a487782f2f218bc6c01e1f366a7e73f002faf08 && git checkout FETCH_HEAD
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
    -DGGML_CUDA=ON -DGGML_CUDA_FA=OFF -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES='86;89;120' \
    -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' \
    && cmake --build build --target llama-server-impl -j 4
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
    /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && cmake --build build --target llama-server ggml-rpc-server -j 4 \
    && mkdir -p /out \
    && cp build/bin/llama-server build/bin/ggml-rpc-server /out/ \
    && find build -name '*.so*' -exec cp -P {} /out/ \;

FROM vllm/vllm-openai:v0.33.0

RUN apt-get update && apt-get install -y --no-install-recommends nfs-common \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models

RUN uv pip install --system 'ray==2.60.0'

WORKDIR /opt/lanllm
COPY lanllm ./lanllm
COPY --from=llama-build /out/ /opt/lanllm/llama/
ENV PATH="/opt/lanllm/llama:${PATH}" \
    LD_LIBRARY_PATH="/opt/lanllm/llama:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}"

ENTRYPOINT ["python3", "-m", "lanllm"]
