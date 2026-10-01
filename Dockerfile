FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    unzip \
    software-properties-common \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    libomp5 \
    vulkan-tools \
    libvulkan1 \
    xdg-user-dirs \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://ratharog.github.io/cumm-spconv/ \
    -r requirements.txt

CMD ["bash", "leaderboard/run_leaderboard.sh"]