#!/bin/bash
set -e  # Exit script on first error
sleep 5 # Wait for the pod to fully start

if [ -n "$RUNPOD_POD_ID" ]; then
    if [ ! -L "examples" ]; then
        echo "📦 Linking examples folder..."
        ln -s /workspace/axolotl/examples .
    fi

    if [ -n "$HF_TOKEN" ]; then
        echo "🔑 Logging in to Hugging Face..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    else
        echo "⚠️ Warning: HF_TOKEN is not set. Skipping Hugging Face login."
    fi

    if [ ! -L "outputs" ]; then
        echo "📦 Linking outputs folder..."
        ln -s /workspace/data/axolotl-artifacts .
        mv axolotl-artifacts outputs
    fi
else
    if [ ! -d "outputs" ]; then
        echo "📦 Creating outputs folder..."
        mkdir outputs
    fi
fi

# check if any env var starting with "AXOLOTL_" is set
if [ -n "$(env | grep '^AXOLOTL_')" ]; then
    echo "⌛ Preparing..."

    if ! python3 configure.py --template config_template.yaml --output config.yaml; then
        echo "❌ Configuration failed!"
    fi
fi

# show message of the day at the Pod logs
cat /etc/motd

# Keeps the container running
sleep infinity
