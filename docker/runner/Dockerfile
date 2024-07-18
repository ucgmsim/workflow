FROM ubuntu:latest

# Requirements for container:
# git (for everything)
# EMOD3D:
# libfftw3-dev
# libpapi-dev
# libopenmpi-dev
# gfortran
# cmake
# Workflow:
# sqlite3 (for NSHMDB)
# gmt + friends (for plotting)
# ffmpeg (for plotting)
# python-is-python3 (for my own sanity)
# python3-pip
RUN apt-get update && apt-get install -y \
    build-essential \
    libfftw3-dev \
    libpapi-dev \
    libopenmpi-dev \
    gfortran \
    git \
    cmake \
    sqlite3 \
    gmt \
    gmt-dcw \
    gmt-gshhg \
    ghostscript \
    gdal-bin \
    graphicsmagick \
    ffmpeg \
    python3-pip \
    python-is-python3 \
    wget
COPY EMOD3D EMOD3D
COPY nshmdb.db nshmdb.db
COPY genslip_velocity_model.vmod genslip_velocity_model.vmod
RUN cd /EMOD3D && \
    git checkout gcc14 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make genslip_v5.4.2 srf2stoch ll2xy xy2ll gen_model_cords
RUN git clone --depth 1 https://github.com/ucgmsim/Velocity-Model.git && \
    cd Velocity-Model && \
    make parallel
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/qcore/ && \
    wget 'https://www.dropbox.com/scl/fi/p5so9irbx43c9g96jr2qr/qcore_resources.tar.xz?rlkey=pgmn69e212c0zx0fauxuikgky&dl=0' -O /tmp/qcore-resources.tar.xz && \
    tar xvf /tmp/qcore-resources.tar.xz -C /usr/local/lib/python3.12/dist-packages/qcore/ && \
    rm /tmp/qcore-resources.tar.xz
ARG anchor cached
RUN pip install \
    "git+https://github.com/ucgmsim/workflow@pegasus" \
    --break-system-packages
ENV PATH="$PATH:/EMOD3D/tools:/Velocity-Model"
