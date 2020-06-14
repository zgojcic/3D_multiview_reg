#!/usr/bin/env bash

DATA_TYPE=$1

function download() {
    if [ ! -d "data" ]; then
        mkdir -p "data"
    fi
    cd data 

    if [ ! -d "eval_data" ]; then
        mkdir -p "eval_data"
    fi
    cd eval_data

    if [ ! -d "3d_match" ]; then
        mkdir -p "3d_match"
    fi
    cd 3d_match		
    
    url="https://share.phys.ethz.ch/~gsg/LMPR/data/"
    
    if [ "$DATA_TYPE" == "raw" ]
    then
        data_set="3d_match_eval_raw.zip"
        echo $url$data_set
    else
        data_set="3d_match_eval_preprocessed.zip"
	echo $url$data_set
    fi
    
    
    wget --no-check-certificate --show-progress "$url$data_set"
    unzip $data_set
    rm $data_set
    cd /../../..


}

function main() {
    if [ -z "$DATA_TYPE" ]; then
        echo "Data type has to be selected! One of [raw, preprocessed]"
	exit 1
    fi

    echo $DATA_TYPE
    if [ "$DATA_TYPE" == "raw" ]  || [ $DATA_TYPE == "preprocessed" ] 
    then
        download
    else
        echo "Wrong data type selected must be on of [raw, preprocessed]."
fi
}

main;
