#!/usr/bin/env bash


SCRIPT="$2"
ARGS="$3"
TAG="$1"
LOGDIR=~/slurm_logs/`hostname`_${TAG}_`date +"%d_%H_%M_%S"`.log

sbatch  -J ${TAG} \
        --exclusive \
        -p lab-ci-student \
	-o ${LOGDIR} \
	${SCRIPT} ${ARGS}


