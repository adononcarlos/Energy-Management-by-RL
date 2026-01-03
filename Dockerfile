FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SINERGYM_ENERGY_PLUS_INSTALLATION_COMPLETE=true

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    python3.12 \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Définir Python 3.12 comme défaut
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Installation d'EnergyPlus 24.2.0a (version officielle Ubuntu 24.04)
RUN wget https://github.com/NREL/EnergyPlus/releases/download/v24.2.0a/EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu24.04-x86_64.sh && \
    chmod +x EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu24.04-x86_64.sh && \
    echo "y" | ./EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu24.04-x86_64.sh --skip-license --prefix=/usr/local/EnergyPlus-24.2.0a && \
    rm EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu24.04-x86_64.sh

ENV PATH="/usr/local/EnergyPlus-24.2.0a:${PATH}"


# Travail dans /workspaces (avec un "s")
WORKDIR /workspaces/energy-rl-project

# Copie requirements.txt depuis ton dossier projet (important !)
COPY requirements.txt .

# Installation du package venv (obligatoire sur Ubuntu 24.04)
RUN apt-get update && apt-get install -y python3.12-venv && rm -rf /var/lib/apt/lists/*

# Création d'un virtualenv pour éviter l'erreur externally-managed
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installation des packages dans le venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir git+https://github.com/AlejandroCN7/opyplus.git@master && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


EXPOSE 6006 8888

# Fix définitif : copie pyenergyplus dans le venv pour contourner le bug d'install
RUN cp -r /usr/local/EnergyPlus-24-2-0/pyenergyplus /opt/venv/lib/python3.12/site-packages/

CMD ["/bin/bash"]