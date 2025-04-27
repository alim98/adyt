#!/bin/bash
echo "Setting up Python path..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
echo "Python path set to: $PYTHONPATH"

echo "Running comparison..."
python compare_mae_methods.py "$@"

echo "Done!" 