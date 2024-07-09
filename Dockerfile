FROM ubuntu:latest

RUN apt update && apt install -y \
    build-essential \
    perl \
    ant \
    openjdk-21-jdk \
    git \
    python3 \
    python3-pip \
    r-base \
    python-is-python3 \
    wget \
    sqlite3 \
    htcondor
RUN wget -O pegasus-5.0.7.tar.gz https://download.pegasus.isi.edu/pegasus/5.0.7/pegasus-5.0.7.tar.gz
RUN tar xvf pegasus-5.0.7.tar.gz
# TAR_OPTIONS is because the package is built in root. Otherwise you
# get an error like:
# https://superuser.com/questions/1435437/how-to-get-around-this-error-when-untarring-an-archive-tar-cannot-change-owner
RUN cd pegasus-5.0.7 \
    && TAR_OPTIONS=--no-same-owner ant dist \ 
    && cd dist/pegasus-5.0.7 \
    && install -d /usr/local/lib/pegasus \
    && install -d /usr/local/share/pegasus \
    && install -Dm755 bin/* /usr/local/bin \
    && install -Dm755 lib/pegasus/*.so /usr/local/lib/pegasus \
    && cp -r lib/pegasus/perl/Pegasus /usr/share/perl/5.38 \
    && cp -r share/pegasus /usr/local/share
RUN rm -rf pegasus-5.0.7*
RUN pip3 install pegasus-wms git+https://github.com/ucgmsim/workflow@pegasus --break-system-packages
RUN pip3 uninstall dataclasses -y --break-system-packages
# YOU NEED TO START condor_master and conder_schedd
