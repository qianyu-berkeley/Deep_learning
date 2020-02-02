#!/bin/bash

for arg in "$@"
do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]
    then
        echo "To run: $0 library log_file_name"
        echo "e.g. > /run_NN_model.sh Parlour run.log"
    fi
done

library=$1
log_file=$2

#if library=='Parlour'
#    echo "Run Parlour library"
python $library/main.py -job "test Parlour" -p $library/config/deep_nn_config_image.json
