#!/bin/bash

echo "Launching Jupyter"
cd /code
mkdir -p output 
nohup jupyter notebook --notebook-dir . --no-browser --port=5000 --ip=0.0.0.0 --allow-root --NotebookApp.token="" --NotebookApp.password="" >output/jupyter.log 2>&1 &

echo "Launching docs server"
mkdir -p docs/_build
nohup python -m http.server --directory docs/_build 9095 >output/docserver.log 2>&1 &

echo "Entering loop"
while sleep 1000; do :; done
