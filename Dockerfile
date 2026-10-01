# 1. Upgrade to CUDA 12.8 to match your +cu128 dependencies
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. Install System Dependencies and Vulkan
# Using native Python 3 (which is 3.10 on Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    unzip \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    libomp5 \
    vulkan-tools \
    vulkan-utils \
    libvulkan1 \
    xdg-user-dirs \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as the default python binary
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 3. Set working directory
WORKDIR /app

# 4. Copy and install Python requirements
COPY requirements.txt .

# We pass the PyTorch cu128 URL here so you don't have to manually edit your requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

# 5. Set Environment Variables
# CARLA_ROOT is still needed by Bench2Drive to locate the simulator executable, 
# but PYTHONPATH is removed because carla is now a global pip package.
ENV CARLA_ROOT=/opt/carla
ENV GPU_RANK=0
ENV PORT=2000

# 6. Execute the evaluation script
CMD ["bash", "leaderboard/scripts/run_evaluation.sh"]