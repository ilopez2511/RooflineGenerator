# GPU-ready base image
FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ---- System packages ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        pkg-config \
        libtool \
        autoconf \
        automake \
        python3 \
        python3-pip \
        python3-dev \
        ca-certificates \
        wget \
        cmake \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Make "python" == "python3"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ---- Python packages (PyTorch, etc.) ----
# NOTE: adjust versions/index-url if needed depending on your CUDA/toolchain.
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
        pandas \
        matplotlib

# Install CUDA-enabled PyTorch + torchvision.
# This combo is an example; adjust if you hit a version mismatch.
RUN python -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ---- Build and install PAPI from source ----
ENV PAPI_PREFIX=/opt/papi

RUN git clone https://github.com/icl-utk-edu/papi.git /tmp/papi && \
    cd /tmp/papi/src && \
    ./configure --prefix=${PAPI_PREFIX} && \
    make -j"$(nproc)" && \
    make install && \
    rm -rf /tmp/papi

# ---- Build and install cyPAPI ----
ENV PAPI_DIR=${PAPI_PREFIX} \
    PAPI_PATH=${PAPI_PREFIX} \
    C_INCLUDE_PATH=${PAPI_PREFIX}/include \
    LIBRARY_PATH=${PAPI_PREFIX}/lib \
    LD_LIBRARY_PATH=${PAPI_PREFIX}/lib:${LD_LIBRARY_PATH}

RUN git clone https://github.com/icl-utk-edu/cyPAPI.git /tmp/cyPAPI && \
    cd /tmp/cyPAPI && \
    make install && \
    rm -rf /tmp/cyPAPI

# ---- Copy your RooflineGenerator project ----
# Assuming Dockerfile is in RooflineGenerator/ on your host
WORKDIR /opt/roofline
COPY . /opt/roofline

# Let Python see your project
ENV PYTHONPATH=/opt/roofline:${PYTHONPATH}

# Default command: run your main handler (change to "profiler.py" if desired)
CMD ["python", "ModelHandler.py"]
