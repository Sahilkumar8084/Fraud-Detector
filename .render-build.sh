#!/usr/bin/env bash
set -o errexit

# Force Python version
PYTHON_VERSION=3.12.6

echo "Installing Python $PYTHON_VERSION..."
pyenv install -s $PYTHON_VERSION
pyenv global $PYTHON_VERSION

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
