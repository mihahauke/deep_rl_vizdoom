FROM ubuntu:16.04

# Cuda 8 with cudnn 5
FROM nvidia/cuda:8.0-cudnn5-devel

# ViZdoom dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng12-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libzmq3-dev \
    nano \
    vim \
    htop \
    nasm \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    zlib1g-dev \
    python3-dev \
    python3 \
    python3-pip \
    language-pack-en-base



# Python3 with pip3
RUN pip3 install pip --upgrade

# Vizdoom and other pip3 packages if needed
RUN git clone https://github.com/mwydmuch/ViZDoom vizdoom

RUN cd vizdoom; pip3 install .

RUN pip3 --no-cache-dir install \
         tensorflow-gpu==1.2.0 \
         opencv-python==3.1.0.3 \
         ruamel.yaml \
         numpy \
         tqdm

RUN echo "export LANG=en_US.UTF-8" > /etc/bash.bashrc
WORKDIR /home/deep_rl_vizdoom
ENV HOME /home
RUN chmod 777 /home




