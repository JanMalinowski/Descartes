FROM ubuntu:20.04


RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m ubuntu
RUN chown -R ubuntu:ubuntu /home/ubuntu/

COPY --chown=ubuntu ./src /home/ubuntu/descartes/src
COPY --chown=ubuntu requirements.txt /home/ubuntu/descartes/
COPY --chown=ubuntu *.sh /home/ubuntu/descartes/

USER ubuntu
RUN cd  /home/ubuntu/descartes/ && pip3 install -r requirements.txt

WORKDIR /home/ubuntu/descartes