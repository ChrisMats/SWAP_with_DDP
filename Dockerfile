
# used these as examples:
# https://github.com/kaust-vislab/python-data-science-project/blob/master/docker/Dockerfile
# https://hub.docker.com/r/anibali/pytorch/dockerfile
# https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

SHELL [ "/bin/bash", "--login", "-c" ]

# install utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         sudo \
         bzip2 \
         libx11-6 \
         git \
         wget \
         ssh-client \
         libjpeg-dev \
         bash-completion \
         libgl1-mesa-dev \
         ffmpeg \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG username=swapuser
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

# switch to that user
USER $USER

# install miniconda
ENV MINICONDA_VERSION latest
# if you want a specific version (you shouldn't) replace "latest" with that, e.g. ENV MINICONDA_VERSION py38_4.8.3

ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
# I named it swap, pick another name if you want
ENV ENV_PREFIX $PROJECT_DIR/env
RUN conda update --name base --channel defaults conda && \
    conda create --name swap python=3.8 -y && \
    conda clean --all --yes

ENV SHELL /bin/bash
# activate the env and install the actual packages you need
# this is not actually activating and we need to use the next line instead
# but then there is a whole other mess, so don't bother with conda envs
RUN conda activate swap
# SHELL ["conda", "run", "-n", "pytorch", "/bin/bash", "-c"]
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN conda install -c conda-forge matplotlib
RUN conda install -c conda-forge scikit-image
RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge easydict
RUN conda install -c conda-forge tqdm
RUN pip install wandb
RUN pip install jupyter jupyterlab
# ENV PATH /home/$USER/.local/bin:$PATH
# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


ENV WANDB_API_KEY 
# RUN wandb login

# git a repo you want to play around with
# RUN git clone https://

# 
WORKDIR /storage
# Start a jupyter notebook
CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0" ]
# or directly login to bash
# CMD ["/bin/bash"]

# run with
# cd <path_to_your_folder_containing_the_Dockerfile>/
# docker build -f Dockerfile -t swap .
# docker run --privileged -p 8866:8888 --gpus all -v /storage:/storage --shm-size=32g swap
# if you setup rootless docker, --privileged is required for accounts intending to use rootful version
