FROM nvcr.io/nvidia/pytorch:23.06-py3

# Instala las dependencias necesarias
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Instala los paquetes de Python que necesitas
RUN python -m pip install --upgrade pip && \
    python -m pip install \
    black \
    ipykernel \
    kaleido>=0.2 \
    lightning>=2.1.9 \
    matplotlib>=3.7 \
    nbformat \
    numpy>=1.22 \
    pandas>=2.0 \
    plotly>=5.18 \
    seaborn>=0.11 \
    scipy>=1.10 \
    torchmetrics>=1.3.0 \
    torchvision>=0.15
