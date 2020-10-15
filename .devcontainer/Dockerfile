FROM jcrist/alpine-conda:4.6.8

RUN /opt/conda/bin/conda update -n base -c defaults conda \
    && /opt/conda/bin/conda install --yes --freeze-installed \
        pytorch torchvision cpuonly  \
        matplotlib \
        ipykernel \
        jupyter -c pytorch \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete
