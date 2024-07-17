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
    python-is-python3
COPY EMOD3D EMOD3D
COPY nshmdb.db nshmdb.db
COPY genslip_velocity_model.vmod genslip_velocity_model.vmod
RUN cd /EMOD3D && \
    git checkout gcc14 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make genslip_v5.4.2 srf2stoch ll2xy xy2ll
RUN git clone --depth 1 https://github.com/ucgmsim/Velocity-Model.git && \
    cd Velocity-Model && \
    make parallel
ARG anchor cached
RUN pip install \
    "git+https://github.com/ucgmsim/workflow@pegasus" \
    --break-system-packages
RUN python -c "from qcore.data import download_data; download_data.download_data()"
ENV PATH="$PATH:/EMOD3D/tools:/Velocity-Model"
