.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


setup: ## setup local virtual environment
	python3.11 -m venv venv && \
	. venv/bin/activate && \
	make install && \
	chmod +x scripts/autorun.sh

install: ## install pip libraries
	pip install --upgrade pip && \
	pip install -r builder/requirements.txt && \
	pip install axolotl --no-deps && \
	pip install peft --no-deps && \
	pip install -r builder/requirements-dev.txt

test: ## run autorun.sh with env vars
	AXOLOTL_BASE_MODEL=TinyLlama/TinyLlama_v1.1 && \
	AXOLOTL_DATASETS='[{"path": "mhenrichsen/alpaca_2k_test","type": "alpaca"}]' && \
	cd scripts && \
	./autorun.sh
