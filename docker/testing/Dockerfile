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
    jq \
    libgmt-dev \
    libfftw3-dev \
    libopenmpi-dev \
    libpapi-dev \
    nodejs \
    python-is-python3 \
    python3-pip \
    sqlite3 \
    wget
COPY EMOD3D EMOD3D
COPY genslip_velocity_model.vmod genslip_velocity_model.vmod
RUN cd /EMOD3D && \
    git checkout gcc14 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make genslip_v5.4.2 srf2stoch ll2xy xy2ll gen_model_cords
RUN wget "https://www.dropbox.com/scl/fi/50kww45wpsnmtf3pn2okz/nshmdb.db?rlkey=4mjuomuevl1963fjwfximgldm&st=o5p83zab&dl=0" -O /nshmdb.db
ENV PATH="$PATH:/EMOD3D/tools"
