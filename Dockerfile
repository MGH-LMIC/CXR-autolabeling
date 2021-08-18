
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

LABEL maintainer="DKim"

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda2
ARG USERNAME=docker
ARG USERID=1000



# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates unzip vim libglib2.0-0 libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDADIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDADIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*
    

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

# Install mamba
RUN /opt/conda2/bin/conda install -y mamba -c conda-forge


ADD ./environment.yml .
COPY --chown=$USERNAME:users ./ .
RUN chmod -R 777 /home/$USERNAME/autolabeling

RUN /opt/conda2/bin/mamba env update --file ./environment.yml &&\
    /opt/conda2/bin/conda clean -tipy

# For interactive shell
RUN /opt/conda2/bin/conda init bash
RUN echo "conda activate lab_env" >> /home/$USERNAME/.bashrc


