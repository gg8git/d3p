FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# some settings to help Docker setup to run smoothly (Learn more: https://chatgpt.com/s/t_68b35fd7e2488191aa97a9092ece474f)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/workspace:${PYTHONPATH}
ENV PYTHONBREAKPOINT=ipdb.set_trace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        curl \
        build-essential \
        libnuma-dev \
        rdma-core \
        ibverbs-providers \
        libibverbs1 \
        vim \
        libgl1 \
        libglib2.0-0 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# pip upgrade
RUN python3 -m pip install --upgrade pip setuptools wheel
    
# torch
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
torch==2.4.0 \
torchvision==0.19.0 \
torchaudio==2.4.0
    
# repo specific dependencies
RUN pip install \
pillow \
tqdm \
numpy \
ipdb \
wandb \
matplotlib \
scikit-learn

# some are pinned right now from importing the Push-T environment
RUN pip install \
diffusers \
scikit-image \
scikit-video \
zarr \
numcodecs \
pygame \
pymunk==6.11.1 \
gym \
gymnasium \
gym-pusht \
shapely \
opencv-python \
gdown

# used to link python as python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/


WORKDIR /workspace
CMD ["/bin/bash"]
