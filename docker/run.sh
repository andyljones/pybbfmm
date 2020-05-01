#!/bin/bash

echo "Launching Jupyter"
cd code
mkdir output 
nohup jupyter notebook --notebook-dir . --no-browser --port=5000 --ip=0.0.0.0 --allow-root --NotebookApp.token="" --NotebookApp.password="" >output/jupyter.log 2>&1 &

echo "Launching $@"
"$@"
