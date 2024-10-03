FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=UTC
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# RUN apt-get update
# # apt-get upgrade -y && \
# # RUN apt-get install -y --force-yes git python3 curl python3-pip

# RUN apt-get clean

ENV MAX_JOBS=24
ENV TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;9.0"
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/cuda/:${PATH}
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install wheel

WORKDIR /backend
COPY ./backend .

RUN pip install .

WORKDIR /app

COPY ./app /app
# Install the Python project using setup.py

RUN apt-get autoremove -y && apt-get clean  && \
    pip cache purge

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
