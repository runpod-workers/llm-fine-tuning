# FROM nvidia/cuda:12.1.0-base-ubuntu22.04 


# RUN ldconfig /usr/local/cuda-12.1/compat/


# # # Set CUDA environment variables
# ENV PATH="/usr/local/cuda/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# RUN apt-get update -y \
#     && apt-get install -y python3-pip git



# # The base image comes with many system dependencies pre-installed to help you get started quickly.
# # Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# # IMPORTANT: The base image overrides the default huggingface cache location.
# # Python dependencies
# Pinned: main-latest moves (it switched to a pip-less uv venv and CUDA 13).
FROM axolotlai/axolotl-cloud:0.21.1-py3.12-cu130-2.14.1

# The base image's python3 is a uv-managed venv without pip. Install into it
# with uv, and add pip so tools that run `python3 -m pip` (such as the Flash
# runtime used by CI) work. No --upgrade: keep the base image's pinned packages.
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "$(command -v python3)" pip && \
    uv pip install --python "$(command -v python3)" -r /requirements.txt

# RUN pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
    
# RUN git clone https://github.com/runpod-workers/axolotl.git && \
#     cd axolotl && \
#     pip install packaging ninja && \
#     pip install --no-build-isolation -e '.[flash-attn,deepspeed]'



# Environment settings
ARG BASE_VOLUME="/runpod-volume"
ENV BASE_VOLUME=$BASE_VOLUME
ENV HF_DATASETS_CACHE="${BASE_VOLUME}/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="${BASE_VOLUME}/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="${BASE_VOLUME}/huggingface-cache/hub"


# Add src files (Worker Template)
COPY src /src

CMD ["python3", "/src/handler.py"]
