FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt-get install sudo -y
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt
RUN python3 -m pip uninstall transformer_engine -y

ARG UNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN echo 'user:docker' | chpasswd
RUN usermod -aG sudo user
USER $UNAME