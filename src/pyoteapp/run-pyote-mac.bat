#!/bin/bash

cd anaconda3

# Create/overwrite python script for starting up pyote
echo "from pyoteapp import pyote" >  run-pyote.py
echo "pyote.main()"               >> run-pyote.py

# Activate the Anaconda3 (base) environment
source activate

# Use python to execute the startup script created above
python run-pyote.py
