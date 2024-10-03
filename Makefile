USE_SUDO := $(shell which docker >/dev/null && docker ps 2>&1 | grep -q "permission denied" && echo sudo)
DOCKER := $(if $(USE_SUDO), sudo docker, docker)
# DIRNAME := $(notdir $(CURDIR))
HAS_NVIDIA_GPU := $(shell which nvidia-smi >/dev/null 2>&1 && nvidia-smi --query --display=COMPUTE >/dev/null 2>&1 && echo ok)
OS := $(shell uname -s)
# Check CUDA availability
ifeq ($(HAS_NVIDIA_GPU),)
$(error CUDA is not available. Please ensure NVIDIA drivers and CUDA are properly installed.)
endif

# Include .env file
include .env
export

# Ensure required environment variables are set
ifndef MODEL_ZIP_URL
$(error MODEL_ZIP_URL is not set. Make sure it's defined in your .env file)
endif
ifndef DATA_ZIP_URL
$(error DATA_ZIP_URL is not set. Make sure it's defined in your .env file)
endif

# Create necessary directories
create-directories:
	mkdir -p backend/model
	mkdir -p app/data

# Download and extract model files
download-model: create-directories
	@if [ ! -f backend/model/model.pt ]; then \
		echo "Downloading model files..."; \
		curl -L "$(MODEL_ZIP_URL)" -o backend/model/model.zip; \
		unzip backend/model/model.zip -d backend/model; \
		rm backend/model/model.zip; \
	else \
		echo "Model files already present."; \
	fi

# Verify model files
verify-model:
	@cd backend/model && \
	for file in attention_output_indication.pkl config.pkl demo.ipynb drug_id.pkl \
				full_graph_split1_eval.pkl gnnexplainer_output_indication.pkl \
				graphmask_output_indication.pkl model.pt name_mapping.pkl \
				node_emb_embed_tsne.pkl node_emb.pkl paths.csv path_viz.csv viz.ipynb; do \
		if [ ! -f "$$file" ]; then \
			echo "Missing file: $$file"; \
			exit 1; \
		fi; \
	done
	@echo "All model files are present."

# Download data files
download-data: create-directories
	@if [ ! -d app/data/dataset ]; then \
		echo "Downloading data files..."; \
		curl -L "$(DATA_ZIP_URL)" -o app/data/data.zip; \
		unzip app/data/data.zip -d app/data; \
		rm app/data/data.zip; \
	else \
		echo "Data files already present."; \
	fi

# Main target to set up everything (downloads and checks run in parallel)
setup: create-directories
	$(MAKE) -j2 download-model download-data
	$(MAKE) verify-model
	@echo "Setup complete."

pull:
	docker-compose pull --parallel

build:
	docker-compose build

up:
	docker-compose up -d

status:
	docker-compose ps

logs:
	docker-compose logs -tf webapp

down:
	docker-compose down --remove-orphans

restart:
	@$(MAKE) down; \
	$(MAKE) up

run: setup pull build up status logs

purge:
	if [ ! -f .env ]; then \
		touch .env; \
	fi
	docker-compose down -v --rmi all --remove-orphans
	if [ -f .env ]; then \
		rm .env; \
	fi

# Declare all targets as .PHONY
.PHONY: create-directories download-model verify-model download-data setup \
        pull build up status logs down restart run purge
