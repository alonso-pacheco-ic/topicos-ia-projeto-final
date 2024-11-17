# Imagen base con PyTorch
FROM nvcr.io/nvidia/pytorch:23.06-py3

# Definir variables de entorno
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV DEBIAN_FRONTEND=noninteractive

# Instalar herramientas esenciales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        git \
        htop \
        libssl-dev \
        net-tools \
        pandoc \
        python3-sphinx \
        tmux \
        tree \
        unzip \
        vim \
        wget \
        sudo && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# Instalar Node.js (opcional si se necesita)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# Crear el usuario y agregar soporte para sudo
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Instalar paquetes de Python, incluyendo PyTorch Lightning
RUN python -m pip install --upgrade pip setuptools pytest && \
    python -m pip install --no-cache-dir \
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
        torchvision>=0.15 \
        wrapt \
        tifffile \
        perlin-noise

# Cambiar el shell para el usuario
RUN chsh --shell /bin/bash $USERNAME

# Exponer puertos necesarios
EXPOSE 8888

# Establecer el usuario por defecto
USER $USERNAME

# Comando por defecto
CMD ["/bin/bash"]