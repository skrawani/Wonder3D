# get the development image from nvidia cuda 11.7
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

LABEL name="Wonder3D" \
    maintainer="Tiancheng <athinkingneal@gmail.com>" \
    lastupdate="2024-01-05" 

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# create workspace folder and set it as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set the timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# update package lists and install git, wget, vim, libgl1-mesa-glx, and libglib2.0-0
RUN apt-get update && \
    apt-get install -y git wget vim libgl1-mesa-glx libglib2.0-0 unzip


# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# initialize conda
RUN conda init bash

# create and activate conda environment
RUN conda create -n wonder3d python=3.8 && echo "source activate wonder3d" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/wonder3d/bin:$PATH


# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get -y install --no-install-recommends \
build-essential \
ca-certificates \
openjdk-8-jdk-headless \
curl \
vim \
&& rm -rf /var/lib/apt/lists/* 


# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip3 --no-cache-dir install mxnet \
                                multi-model-server \
                                sagemaker-inference \
                                retrying





# clone the repository
RUN git clone https://github.com/xxlong0/Wonder3D.git && \
    cd /workspace/Wonder3D 

# Copy entrypoint script to the image
COPY dockerd-entrypoint.py /workspace/Wonder3D/dockerd-entrypoint.py
RUN chmod +x /workspace/Wonder3D/dockerd-entrypoint.py

COPY model_handler.py /workspace/Wonder3D

# change the working directory to the repository
WORKDIR /workspace/Wonder3D

# install pytorch 1.13.1 and torchvision
RUN pip install -r docker/requirements.txt --no-cache-dir 

# install the specific version of nerfacc corresponding to torch 1.13.0 and cuda 11.7, otherwise the nerfacc will freeze during cuda setup
RUN pip install nerfacc==0.3.3 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu117.html

# install tiny cuda during docker setup will cause error, need to install it manually in the container
# RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]

