FROM pytorch/pytorch

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive
    
RUN conda config --add channels conda-forge \
    && conda update -n base -c defaults conda \
    # maybe get a full environment.yml file	
    # for more precision later on	
    && conda install -y matplotlib ipykernel jupyter \
    && rm -rf /opt/conda/pkgs/*