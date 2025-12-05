# Base image with CUDA + dev tools
FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04


# 1) System packages

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        python3 \
        python3-pip \
        python3-dev \
        pkg-config \
        libtool \
        autoconf \
        automake \
        libhwloc-dev && \
    rm -rf /var/lib/apt/lists/*

# Make `python` == python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1


# 2) Python packages

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
        pandas \
        matplotlib \
        numpy

# PyTorch + torchvision for CUDA 12.3.x
RUN python -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121


# 3) Make sure CUDA / CUPTI libs are on the path

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}


# 4) Build & install PAPI with CUDA component

RUN git clone https://github.com/icl-utk-edu/papi.git /tmp/papi && \
    cd /tmp/papi/src && \
    ./configure \
      --prefix=/usr/local \
      --with-cuda=${CUDA_HOME} \
      --with-cuda-lib=${CUDA_HOME}/lib64 \
      --with-cuda-include=${CUDA_HOME}/include && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf /tmp/papi


# 5) Build & install cyPAPI

# Install cyPAPI against the PAPI we just built into /usr/local
RUN git clone https://github.com/icl-utk-edu/cyPAPI.git /tmp/cyPAPI && \
    cd /tmp/cyPAPI && \
    PAPI_PREFIX=/usr/local python -m pip install . && \
    ldconfig && \
    rm -rf /tmp/cyPAPI


WORKDIR /opt/roofline
COPY . /opt/roofline
# Default command: run the model handler
# ENTRYPOINT ["python", "ModelHandler.py"]
