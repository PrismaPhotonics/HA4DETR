FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create build venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel build

# Install PyTorch 2.8 (CUDA 12.4)
RUN pip install torch==2.8.0 #--index-url https://download.pytorch.org/whl/cu124

# Copy your project
WORKDIR /workspace

# Build the wheel
#RUN python -m build


