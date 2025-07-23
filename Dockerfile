FROM axolotlai/axolotl-cloud:main-latest AS builder

# Set environment variables to reduce size
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /requirements.txt

# Create a new stage with minimal files
FROM axolotlai/axolotl-cloud:main-latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy installed packages from builder stage - selectively to save space
COPY --from=builder /usr/local/lib/python3.11/site-packages/runpod /usr/local/lib/python3.11/site-packages/runpod
COPY --from=builder /usr/local/lib/python3.11/site-packages/huggingface_hub /usr/local/lib/python3.11/site-packages/huggingface_hub
COPY --from=builder /usr/local/lib/python3.11/site-packages/typing_extensions* /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/lib/python3.11/site-packages/pydantic* /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/lib/python3.11/site-packages/hf_transfer /usr/local/lib/python3.11/site-packages/hf_transfer
COPY --from=builder /usr/local/lib/python3.11/site-packages/numpy /usr/local/lib/python3.11/site-packages/numpy
COPY --from=builder /usr/local/bin/runpod* /usr/local/bin/

# Clean up to save space
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    # Remove unnecessary CUDA files to save space
    rm -rf /usr/local/cuda/samples /usr/local/cuda/doc /usr/local/cuda/extras && \
    # Remove unnecessary Python cache files
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /root -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Clean up NVIDIA CUDA libraries in both potential locations to save space
RUN rm -rf /usr/local/cuda/lib64/libcudnn* \
           /usr/local/cuda/lib64/libnvjpeg* \
           /usr/local/cuda/lib64/libnvjitlink* \
           /usr/local/cuda/lib64/libcufft* \
           /usr/local/cuda/lib64/libcurand* \
           /usr/local/cuda/lib64/libnvgraph* \
           /usr/local/cuda/lib64/libnpp* \
           /usr/local/cuda/lib64/libnvrtc* \
           /usr/local/cuda/lib64/libnvToolsExt* \
           /usr/local/cuda/lib64/libcupti* 2>/dev/null || true && \
    # Keep only essential CUDA libraries
    mkdir -p /tmp/essential_libs && \
    # Save essential libraries before cleanup
    cp -a /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so* /tmp/essential_libs/ 2>/dev/null || true && \
    cp -a /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/nvidia/cusolver/lib/libcusolver.so* /tmp/essential_libs/ 2>/dev/null || true && \
    # Clean up NVIDIA libraries in miniconda environment
    rm -rf /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/nvidia/*/lib/* 2>/dev/null || true && \
    # Restore essential libraries
    mkdir -p /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/nvidia/cublas/lib/ 2>/dev/null || true && \
    mkdir -p /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/nvidia/cusolver/lib/ 2>/dev/null || true && \
    cp -a /tmp/essential_libs/libcublas.so* /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/nvidia/cublas/lib/ 2>/dev/null || true && \
    cp -a /tmp/essential_libs/libcusolver.so* /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/nvidia/cusolver/lib/ 2>/dev/null || true && \
    rm -rf /tmp/essential_libs

# Environment settings
ARG BASE_VOLUME="/runpod-volume"
ENV BASE_VOLUME=$BASE_VOLUME \
    HF_DATASETS_CACHE="${BASE_VOLUME}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_VOLUME}/huggingface-cache/hub" \
    TRANSFORMERS_CACHE="${BASE_VOLUME}/huggingface-cache/hub"

# Add src files (Worker Template)
COPY src /src

# Copy test_input.json if it exists (for RunPod SDK automated testing)
COPY test_input.json /

CMD ["python3", "/src/handler.py"]