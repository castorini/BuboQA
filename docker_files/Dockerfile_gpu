FROM nvidia/cuda:9.0-base-ubuntu16.04

LABEL maintainer="Salman Mohammed"

ENV LC_ALL C

# Setting CUDA paths
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu
ENV CUDA_HOME /usr/local/cuda-9.0


# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        vim \
        strace \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev python3-pip python3.6-venv python3.6-tk python3-setuptools locales
RUN apt-get install -y git


# Python should link to Python 3.6
RUN ln -s -f /usr/bin/python3.6 /usr/bin/python3
RUN ln -s -f /usr/bin/python3 /usr/bin/python

RUN pip3 install pip --upgrade
RUN pip3 install wheel


# Required Python packages
RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        ipython \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        PyYaml \
        sklearn-pandas \
        feather-format \
        spacy \
        nltk \
        fuzzywuzzy

# Install Spacy model and NLTK data
RUN python -m spacy download en
RUN python -m nltk.downloader all

# Install PyTorch version 0.4 for Cuda 9.0 & Python 3.6
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl

# Install torchtext 0.2.3
RUN pip3 install torchtext==0.2.3


WORKDIR /code
COPY ./ /code

