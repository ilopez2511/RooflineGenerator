FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# symlink nvcc so PAPI's tests find it
RUN ln -s /usr/local/cuda/bin/nvcc /usr/bin/nvcc || true \
    && ln -s /usr/local/cuda/bin/nvcc /bin/nvcc || true

RUN pip3 install --no-cache-dir \
        cython>=3.0.0 \
        numpy \
        matplotlib \
        pandas \
        seaborn \
        pkgconfig

ARG PAPI_GIT_REF=a239b884b2fde0da425ad66b19c5836590b64e40

RUN git clone https://github.com/icl-utk-edu/papi.git /tmp/papi \
    && cd /tmp/papi \
    && git checkout "$PAPI_GIT_REF" \
    && cd src \
    && env PAPI_CUDA_ROOT=/usr/local/cuda ./configure \
         --prefix=/opt/papi --with-components="cuda" \
    # first build everything (library + component tests)
    && make -j$(nproc) \
    # then install the built library into /opt/papi
    && make install \
    && rm -rf /tmp/papi

ENV PAPI_DIR=/opt/papi \
    PAPI_PATH=/opt/papi \
    PAPI_CUDA_ROOT=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/papi/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH}

# install cyPAPI
RUN git clone https://github.com/icl-utk-edu/cyPAPI.git /tmp/cyPAPI \
    && cd /tmp/cyPAPI \
    && export PAPI_DIR=/opt/papi PAPI_PATH=/opt/papi \
             C_INCLUDE_PATH=/opt/papi/include LIBRARY_PATH=/opt/papi/lib \
             LD_LIBRARY_PATH=/opt/papi/lib:${LD_LIBRARY_PATH} \
    && make install \
    && rm -rf /tmp/cyPAPI

# copy your project files into the image
COPY . /opt/roofline/
ENV PYTHONPATH=/opt/roofline:$PYTHONPATH
WORKDIR /opt/roofline
CMD ["python3", "profiler.py"]
