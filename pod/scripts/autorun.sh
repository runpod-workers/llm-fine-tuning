#!/bin/bash
set -e  # Exit script on first error

if [ ! -L "examples" ]; then
    echo "📦 Linking examples..."
    ln -s /workspace/axolotl/examples .
fi

if [ -n "$HF_TOKEN" ]; then
    echo "🔑 Logging in to Hugging Face..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "⚠️ Warning: HF_TOKEN is not set. Skipping Hugging Face login."
fi

echo "⌛ Preparing..."
if ! python3 configure.py --template config_template.yaml --output config.yaml; then
    echo "❌ Configuration failed!"
    sleep infinity  # Keeps the container running for inspection
fi

echo "🚀 Training..."
axolotl train config.yaml || { echo "❌ Training failed. Exiting."; sleep infinity; }

echo "✅ Training complete. Keeping container alive..."
sleep infinity
