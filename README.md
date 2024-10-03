# TxGNN Restful API - FastAPI with CUDA Support and Docker Compose

This repository provides the implementation of a RESTful API that exposes TxGNN's functionality for zero-shot therapeutic predictions and explanations using geometric deep learning (GNN). This API is deployed using **FastAPI**, with **CUDA** support for GPU acceleration, and containerized using **Docker**. Additionally, the repository includes a `Makefile`, `docker-compose.yml`, and `.env` file for convenient management and deployment of the application.

## TL;DR - Quickstart

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/txgnn-rest-api.git
    cd txgnn-rest-api
    cp .env.example .env
    ```
2. Run the application using Make
    ```bash
    make run
    ```
3. Access the API Once the services are up, the API will be [locally accessible at port 8883](http://localhost:8883).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Installation](#step-by-step-installation)
  - [Docker Compose Setup](#docker-compose-setup)
- [Makefile Commands](#makefile-commands)
- [API Endpoints](#api-endpoints)
- [CUDA Support](#cuda-support)
- [Citation](#citation)
- [License](#license)

## Overview

TxGNN is a graph neural network (GNN) trained on a knowledge graph of diseases and therapeutic candidates. This REST API allows zero-shot therapeutic predictions and explanations using pre-trained models, supporting CUDA-enabled GPU acceleration via Docker.

## Features
- **CUDA-enabled**: GPU-accelerated predictions for faster performance.
- **FastAPI**: A modern, fast web framework for building RESTful APIs.
- **Docker**: Containerized deployment for easy setup and scalability.

---

## Setup and Installation

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Make](https://www.gnu.org/software/make/)

---

### Docker Compose Setup

The repository includes a `docker-compose.yml` file for managing the Docker container setup. This setup ensures a smooth environment for the API with CUDA support.

**Environment variables**:
```env
[MODEL_ZIP_URL](https://drive.usercontent.google.com/download?id=1fxTFkjo2jvmz9k6vesDbCeucQjGRojLj&export=download&authuser=0&confirm=t&uuid=4a7bcb2a-7391-445e-86a6-060b7503d6c9&at=AN_67v0G7YEwLsFuqX52PVDMecVP%3A1727988066581)=https://drive.usercontent.google.com/download?id=1fxTFkjo2jvmz9k6vesDbCeucQjGRojLj&export=download&authuser=0&confirm=t&uuid=4a7bcb2a-7391-445e-86a6-060b7503d6c9&at=AN_67v0G7YEwLsFuqX52PVDMecVP%3A1727988066581
[DATA_ZIP_URL](https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/IXA7BM/1805e679c4c-72137dbedbf1?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27kg.csv&response-content-type=text%2Fcsv&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241003T204240Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20241003%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=287af25ab7dc3c691c7aa9fcca093c9652a7814c1e6e5c314829e17a6504588a)=https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/IXA7BM/1805e679c4c-72137dbedbf1?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27kg.csv&response-content-type=text%2Fcsv&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241003T204240Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20241003%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=287af25ab7dc3c691c7aa9fcca093c9652a7814c1e6e5c314829e17a6504588a
```

1. **Clone the repository**
    ```bash
    git clone https://github.com/healthecosystem/TxGNN.git
    cd txgnn-rest-api
    cp .env.example .env
    ```
2. **Pull**, **build**, and **run** the services Use the following commands from the `Makefile` to pull the required images, build, and run the container:
    ```bash
    make run
    ```
3. Access the API Once the services are up, the API will be [locally accessible at port 8883](http://localhost:8883).

## Makefile Commands

The `Makefile` provides a set of commands to simplify the development and deployment of the Docker-based FastAPI app. Here are the available commands:

- **`make copy-env`**: Copies the example environment variables file `.env.example` to `.env`.

- **`make pull`**: Pulls the latest Docker images defined in the `docker-compose.yml` file, using parallel processing.

- **`make build`**: Builds the Docker containers as defined in the `docker-compose.yml` file.

- **`make up`**: Starts the Docker containers in detached mode.

- **`make status`**: Displays the status of the running Docker containers.

- **`make logs`**: Shows the logs from the FastAPI application for debugging purposes.

- **`make down`**: Stops the running containers and removes any orphaned containers.

- **`make restart`**: Restarts the Docker containers by first stopping them and then starting them again.

- **`make run`**: Executes a series of commands to set up the environment and start the application. This command performs the following steps:
  1. Copies the environment variables file.
  2. Pulls the required Docker images.
  3. Builds the Docker containers.
  4. Starts the containers.
  5. Displays the status of the containers.
  6. Sets up any necessary configurations.
  7. Restarts the containers.
  8. Shows the logs.

- **`make purge`**: Completely removes all Docker containers, volumes, and images related to the project. This command also handles the `.env` file:
  1. Creates an empty `.env` file if it does not exist.
  2. Stops all containers and removes volumes, images, and orphaned containers.
  3. Deletes the `.env` file if it exists.

---

## API Endpoints

### Health Check
- **Endpoint**: `/healthz`
- **Method**: `GET`
- **Description**: Returns the health status of the API.

### Predict Drug Replacement
- **Endpoint**: `/predict`
- **Method**: `GET`
- **Parameters**: `disease (str)`
- **Description**: Predicts a drug replacement for a given disease.

### Explain Drug Replacement
- **Endpoint**: `/explain`
- **Method**: `GET`
- **Parameters**: `disease (str)`, `drug (str)`
- **Description**: Explains why a drug is recommended as a replacement for the specified disease.

---

## CUDA Support

This API is designed to leverage CUDA for enhanced performance on compatible GPU hardware. Ensure that your environment meets the CUDA installation requirements and that the appropriate NVIDIA drivers are installed.

For optimal performance, confirm that the container is run with the `--gpus all` option to utilize all available GPUs.

---

## Citation

If you use TxGNN or this API in your research, please cite the relevant papers and resources associated with TxGNN.

[MedRxiv preprint](https://www.medrxiv.org/content/10.1101/2023.03.19.23287458)

```
@article{huang2023zeroshot,
  title={Zero-shot Prediction of Therapeutic Use with Geometric Deep Learning and Clinician Centered Design},
  author={Huang, Kexin and Chandak, Payal and Wang, Qianwen and Havaldar, Shreyas and Vaid, Akhil and Leskovec, Jure and Nadkarni, Girish and Glicksberg, Benjamin and Gehlenborg, Nils and Zitnik, Marinka},
  journal = {medRxiv},
  doi = {10.1101/2023.03.19.23287458},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={}
}
```


---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
