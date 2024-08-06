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
    cmake \
    ffmpeg \
    gdal-bin \
    gfortran \
    ghostscript \
    git \
    gmt \
    gmt \
    gmt-dcw \
    gmt-gshhg \
    graphicsmagick \
    libfftw3-dev \
    libopenmpi-dev \
    libpapi-dev \
    python-is-python3 \
    python3-pip \
    sqlite3 \
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
RUN wget "https://www.dropbox.com/scl/fi/xomwzo5xagrxzbdmm7z7w/Velocity-Model-1.0.0.tar.gz?rlkey=0vc6kevs851qos39kkmd4ffol&dl=0" -O /Velocity-Model.tar.gz && \
    tar xvf /Velocity-Model.tar.gz && \
    mv Velocity-Model-1.0.0 Velocity-Model && \
    cd Velocity-Model && \
    make parallel && \
    rm /Velocity-Model.tar.gz
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/qcore/ && \
    wget 'https://www.dropbox.com/scl/fi/p5so9irbx43c9g96jr2qr/qcore_resources.tar.xz?rlkey=pgmn69e212c0zx0fauxuikgky&dl=0' -O /tmp/qcore-resources.tar.xz && \
    tar xvf /tmp/qcore-resources.tar.xz -C /usr/local/lib/python3.12/dist-packages/qcore/ && \
    rm /tmp/qcore-resources.tar.xz
ARG anchor cached
RUN pip install \
    "git+https://github.com/ucgmsim/workflow@pegasus" \
    --break-system-packages
ENV PATH="$PATH:/EMOD3D/tools:/Velocity-Model"
