#!/bin/bash

script=$(./load/get_activation_script.py "$@")
if [ $? -eq 0 ]
then
  echo "sourcing: ${script}"
  source ${script}
fi

