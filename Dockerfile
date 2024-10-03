FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=UTC
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# RUN apt-get update
# # apt-get upgrade -y && \
# # RUN apt-get install -y --force-yes git python3 curl python3-pip

# RUN apt-get clean

ENV MAX_JOBS=24
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/cuda/:${PATH}
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install wheel
# && \
# TODO TensorRT lean
# python3 -m pip install --upgrade tensorrt_lean
# python3 -m pip install --upgrade tensorrt_dispatch
# python3 -m pip install --upgrade tensorrt


WORKDIR /backend
COPY ./backend .

RUN pip install .

WORKDIR /app

COPY ./app /app
# Install the Python project using setup.py

RUN apt-get autoremove -y && apt-get clean  && \
    pip cache purge

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
