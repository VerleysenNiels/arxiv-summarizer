FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Brussels
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y tzdata

RUN apt update -y
RUN apt install python3-dev python3-pip python3-venv -y

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl \
    git \
    make \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    unzip \
    zip \
  && apt-get clean

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install numpy
RUN pip3 install transformers
RUN pip3 install datasets
RUN pip3 install evaluate
RUN pip3 install nltk
RUN pip3 install gradio
RUN pip3 install tensorboard
RUN pip3 install rouge_score