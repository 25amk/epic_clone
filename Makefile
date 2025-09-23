#
# Description: Makefile for the dev lifecycle of the epic-app project
#

APPNAME := epic
VENV := .venv
SYS_PY := python3.12
PY := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip3
REQUIREMENTS := requirements.txt
DOCKER_BINARY := docker
DOTENV := .env
JUPYTER_HOST := 127.0.0.1
JUPYTER_PORT := 8888
JUPYTER_NBDIR := nb
SH := bash
SHELL := bash
SITE_PROFILE ?= localdev
ENVS_SCRIPT := $(SITE_PROFILE)-envs.sh

help:
	-@echo ""
	-@echo "Usage: make <target>"
	-@echo ""
	-@echo "[development environment]"
	-@echo "  init / fini: create / destroy local dev environment"
	-@echo "  chat: Run the chat web UI"
	-@echo "  up/down: Start(up) stop(down) background daemons"
	-@echo "  shell: Jump into a shell with the virtual environemnt activated"
	-@echo "  ipython: Jump into a python shell with the virtual environemnt activated"
	-@echo "  jupyter: Start a local jupyter lab server as a shell"
	-@echo ""
	-@echo "[Environment setup]"
	-@echo "  env / env-fini: fetch remote credentials from ./site/***"
	-@echo "  generate-data: generate synthetic data for the repository"
	-@echo "  fetch-models: prefetch models to local cache from Hugging Face"
	-@echo ""
	-@echo "[development cycle]"
	-@echo "  test: Run unit tests"
	-@echo "  integration-test: Run integration tests"
	-@echo "  pipeline-test: Run tests for the pipelines"
	-@echo "  test-all: Run all tests"
	-@echo ""

.PHONY: help all freeze init fini
all: help

#
# Virtual environment management
#
# Fulfills the requirements from the requirements.lock.txt file but falls
# back to the requirements.txt file if the lock file does not exist.
#
# The requirements.lock.txt file is updated with the freeze target and can be
# updated by erasing the requirements.lock.txt file first, run `make fini` and then running
# `make init` again.
#
init:
	@echo "Initializing..."
	@echo "Creating virtual environment..."
	@$(SYS_PY) -m venv $(VENV)
	@echo "Activating virtual environment..."
	@. $(VENV)/bin/activate
	@echo "Installing requirements..."
	@$(PIP) install -U pip && $(PIP) install -r $(REQUIREMENTS)
	@$(PY) -m ipykernel install --user --name=$(APPNAME)
	@$(PIP) install -e .
	@mkdir -p ./.metaflow
	@echo "Installing node"
	@curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
	@source ~/.nvm/nvm.sh && nvm install v20
	@source ~/.nvm/nvm.sh && nvm use v20 && cd custom_elements && npm install && npm run build
	@echo "Initialization complete."

fini:
	@echo "Removing virtual environment..."
	-@$(VENV)/bin/jupyter kernelspec remove $(APPNAME)
	-@rm -rf $(VENV)
	@echo "Finalization complete."


.PHONY: env env-fini
env:
	@echo "Setting up environment (.env) using site profile $(SITE_PROFILE)"
	@source $(VENV)/bin/activate && ./site/$(ENVS_SCRIPT) setup
	@rm -rf ./site/_current.sh && ln -s $(ENVS_SCRIPT) ./site/_current.sh

env-fini:
	@echo "Tearing down environment (.env)"
	@if test -e ./site/_current.sh ; then source $(VENV)/bin/activate && ./site/_current.sh teardown ; fi
	@rm -rf ./site/_current.sh


.PHONY: build push
build:
	@echo "Building using docker using the $(DOTENV) environment file"
	@source $(DOTENV) && $(DOCKER_BINARY) build -t $${REGISTRY}/$${APPIMAGE}:${APPTAG} .

push:
	@echo "Pushng the image using the $(DOTENV) environment file"
	@source $(DOTENV) && $(DOCKER_BINARY) push -t $${REGISTRY}/$${APPIMAGE}:${APPTAG}

.PHONY: build-js
build-js:
	@source ~/.nvm/nvm.sh && nvm use v20 && cd custom_elements && npm run build

.PHONY: generate-data fetch-data fetch-models
generate-data:
	@echo "Generating synthetic data"
	mkdir -p .data
	./scripts/generate_frontier_job_data.py --start=2024-01-01T00:00:00+00:00 --end=2024-02-01T00:00:00+00:00 --out .data/private/ornl/frontier/jobsummary

fetch-data:
	mkdir -p .data/shared/riken/fugaku
	curl https://zenodo.org/api/records/11467483/files-archive -o .data/shared/riken/fugaku/fugaku.zip
	unzip .data/shared/riken/fugaku/fugaku.zip -d .data/shared/riken/fugaku
	rm .data/shared/riken/fugaku/fugaku.zip

fetch-models:
	@echo ""
	@echo "Fetching Hugging Face Models (pre-download)"
	@echo ""
	@echo "- SQL model"
	$(VENV)/bin/huggingface-cli download defog/llama-3-sqlcoder-8b
	@echo ""
	@echo "- General Purpose Instruct model - Llama3.1-8B"
	$(VENV)/bin/huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct
	@echo ""
	@echo "- General Purpose Instruct model - Llama3.2-3B"
	$(VENV)/bin/huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
	@echo ""
	@echo "- RAG Embedding model"
	$(VENV)/bin/huggingface-cli download mixedbread-ai/mxbai-embed-large-v1
	@echo ""
	@echo "- RAG Reranker model"
	$(VENV)/bin/huggingface-cli download mixedbread-ai/mxbai-rerank-large-v1


.PHONY: chat shell ipython up down jupyter
chat:
	@echo "Starting the chat UI"
	@source $(VENV)/bin/activate && epc chat


shell:
	@echo "Jumping into shell"
	@source $(VENV)/bin/activate && $(SH)

ipython:
	@echo "Jumping into an ipython shell"
	@source $(VENV)/bin/activate && ipython

up:
	@echo "up not implemented yet"

down:
	@echo "down not implemented yet"

jupyter:
	@echo "Running the jupyter server"
	@source $(VENV)/bin/activate && jupyter lab --ip $(JUPYTER_HOST) --port $(JUPYTER_PORT) --notebook-dir=$(JUPYTER_NBDIR)


.PHONY: test integration-test pipeline-test test-all
test:
	@$(VENV)/bin/pytest -m unit

integration-test:
	@$(VENV)/bin/pytest -m integration

pipeline-test:
	@$(VENV)/bin/pytest -m pipeline

test-all:
	@$(VENV)/bin/pytest
