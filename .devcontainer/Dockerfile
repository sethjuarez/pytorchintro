FROM pytorch/pytorch
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive
RUN conda config --add channels conda-forge \
    # maybe get a full environment.yml file
    # for more precision later on
    && conda install -y matplotlib jupyter \
    && rm -rf /opt/conda/pkgs/*

