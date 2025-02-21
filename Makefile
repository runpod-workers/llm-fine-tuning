.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


setup: ## setup the virtual environment
	python3.11 -m venv venv && \
	. venv/bin/activate && \
	make install

install: ## install pip libraries
	pip install --upgrade pip && \
	pip install -r builder/requirements.txt && \
	pip install axolotl --no-deps && \
	pip install peft --no-deps && \
	pip install -r builder/requirements.dev.txt

test: ## run src/configure.py
	AXOLOTL_BASE_MODEL=m-a-p/FineFineWeb && \
	AXOLOTL_DATASETS='[{"path": "datasets/alpaca_data.json","type": "alpaca","field_map": {"prompt": "instruction","response": "output"}}]' && \
	AXOLOTL_PEFT='{"task_type": "CAUSAL_LM", "r": 32, "lora_alpha": 16}' && \
	cd src && \
	./run.sh
