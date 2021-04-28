#!/bin/bash

./load/get_activation_script.py "$@"

if [ -f ./load/tmp_script_name ]
then
  script=$(cat ./load/tmp_script_name)
  rm ./load/tmp_script_name
  echo "sourcing: ${script}"
  source ${script}
fi

