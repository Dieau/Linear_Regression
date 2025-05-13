#!/bin/bash

set -e

echo "Checking Python version..."
if ! python3 -c 'import sys; assert sys.version_info >= (3, 8)' &>/dev/null; then
    echo "ERROR: Python 3.8 or higher is required."
    exit 1
fi

echo "Creating virtual environment..."
python3 -m venv .venv

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    ACTIVATE_SCRIPT=".venv/Scripts/activate"
else
    ACTIVATE_SCRIPT=".venv/bin/activate"
fi

echo "Installing dependencies..."
source "$ACTIVATE_SCRIPT"
pip install --upgrade pip
pip install -e ".[dev]"

echo "Verifying installation..."
python -c "import training, prediction" && echo "Import test passed."

echo "Checking for dependency conflicts..."
if ! pip check; then
    echo "⚠️  Warning: Dependency conflicts detected. Run 'pip check' for details."
fi

echo -e "\n🎉 Setup complete! Virtual environment is now active.\n"
echo "To deactivate, run:"
echo "  deactivate"
echo "To reactivate later, run:"
echo "  source \"$ACTIVATE_SCRIPT\""

exec "$SHELL"