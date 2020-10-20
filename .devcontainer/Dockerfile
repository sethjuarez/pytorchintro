FROM python:3.8-slim

RUN export DEBIAN_FRONTEND=noninteractive && \
    pip install --no-cache-dir \
                    matplotlib \
                    ipykernel \
                    jupyter \
                    torch==1.6.0+cpu \
                    torchvision==0.7.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html