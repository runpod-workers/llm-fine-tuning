#!/bin/bash

echo "🔧 Configuring..."
python3 /src/configure.py

if [ -n "$HF_TOKEN" ]; then
    echo "🔑 Logging in to Hugging Face..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "⚠️ Warning: HF_TOKEN is not set. Skipping Hugging Face login."
fi

# Run CMD
exec "$@"
