FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt update -y && apt install -y \
    git
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    vim \
    swig

# Copy files from host to the image.
COPY requirements.txt /tmp/requirements.txt

# Install python package, remove copied file and cache.
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

# Directory settings for login
WORKDIR /working
RUN chmod 777 /working
