#!/bin/bash

script=$(./load/get_activation_script.py "$@")

echo "sourcing: ${script}"
source ${script}
