#!/usr/bin/env bash

NAME="mgr_`hostname`"
TAG="mgr"


docker build -t ${TAG} .

nvidia-docker run \
        --user=`id -u`:`id -g`\
       --net=host \
       --name ${NAME} \
       -v `pwd`:/home \
       --rm\
        ${TAG} \
        "$@"
