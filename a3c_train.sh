#!/usr/bin/env bash

ERLOG_DIR="erlogs"
date=`date +%d-%m-%y_%H:%M`
ERLOG_PATH=${ERLOG_DIR}/erlog_${date}.log
mkdir -p ${ERLOG_DIR}


python a3c_train.py $@ 2>${ERLOG_PATH}