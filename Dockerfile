FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SINERGYM_ENERGY_PLUS_INSTALLATION_COMPLETE=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    python3.10 \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 6006 8888

CMD ["/bin/bash"]