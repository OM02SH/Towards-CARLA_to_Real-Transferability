FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    unzip \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libusb-1.0-0 \
    libgomp1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libxfixes3 \
    libxi6 \
    libxcb1 \
    libglu1-mesa \
    libsdl2-2.0-0 \
    libudev1 \
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    libomp5 \
    vulkan-tools \
    libvulkan1 \
    xdg-user-dirs \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://ratharog.github.io/cumm-spconv/ \
    -r requirements.txt

RUN pip install --no-cache-dir \
    https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8/cumm_cu128-0.8.2-cp310-cp310-linux_x86_64.whl \
    https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8/spconv_cu128-2.3.8-cp310-cp310-linux_x86_64.whl

COPY OpenPCDet /opt/OpenPCDet
RUN pip install --no-cache-dir -e /opt/OpenPCDet

ENV PYTHONPATH=/app:/opt/carla/PythonAPI/carla

CMD ["bash", "-c", "cd /app/leaderboard && bash run_leaderboard.sh"]