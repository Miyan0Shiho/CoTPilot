#!/bin/bash

echo "🚀 Installing CoT-Pilot Environment..."

# 1. Core Dependencies
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# 2. EvoPrompt Dependencies
if [ -d "third_party/EvoPrompt" ]; then
    echo "📦 Installing EvoPrompt dependencies..."
    cd third_party/EvoPrompt
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "   No requirements.txt found for EvoPrompt."
    fi
    cd ../..
else
    echo "⚠️  EvoPrompt directory not found!"
fi

# 3. OpenCompass Dependencies
if [ -d "third_party/opencompass" ]; then
    echo "📦 Installing OpenCompass dependencies..."
    cd third_party/opencompass
    # Install in editable mode to respect local changes
    pip install -e .
    cd ../..
else
    echo "⚠️  OpenCompass directory not found!"
fi

echo "✅ Installation Complete! You can now run experiments using 'python main.py'."
