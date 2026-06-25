# Bootstrap Dockerfile
FROM ubuntu:latest

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install build dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        curl \
        gfortran \
        git \
        pkg-config \
        libfftw3-dev \
        libpapi-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        libopenblas-openmp-dev \
        libhdf5-dev \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
        libexpat1-dev \
        libgdbm-dev \
        libgdbm-compat-dev \
        libncurses5-dev \
        libncursesw5-dev \
        uuid-dev \
        libnss3-dev \
        libedit-dev \
        libuuid1 \
        libdb5.3-dev \
        lzma \
        tcl-dev \
        gdb \
        lcov \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create software directories (removed uv directory)
RUN mkdir -p /sw/cargo /sw/rust /sw/python /sw/venv

# Python installation
ARG PYTHON_VERSION=3.13.12
RUN curl -o /tmp/Python-${PYTHON_VERSION}.tar.xz \
        https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    cd /tmp && \
    tar xf Python-${PYTHON_VERSION}.tar.xz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --prefix=/sw/python --enable-optimizations && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    rm -rf Python-${PYTHON_VERSION}*

# Set Environment Paths
ENV RUSTUP_HOME=/sw/rust
ENV CARGO_HOME=/sw/cargo
ENV VIRTUAL_ENV="/sw/venv"
# Added /sw/python/bin to path early to ensure we use the compiled python
ENV PATH="$VIRTUAL_ENV/bin:/sw/python/bin:/sw/cargo/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Setup Virtual Environment and Install via pip
ARG WORKFLOW_BRANCH=76cc90e8fab9f4d22313ca7979eea64ddd6761e2
ARG VM_BRANCH=master
RUN python3 -m venv $VIRTUAL_ENV && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "git+https://github.com/lispandfound/nzcvm@${VM_BRANCH}" "git+https://github.com/ucgmsim/workflow@${WORKFLOW_BRANCH}" --no-binary=h5py && \
    chmod -R a+rX /sw
