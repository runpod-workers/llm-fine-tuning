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

