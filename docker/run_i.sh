#!/usr/bin/env bash

NAME="mgr_`hostname`"
image_tag=mgr

docker run \
       --user=`id -u`:`id -g`\
       --net=host \
       -it \
       --name ${NAME} \
       -v `pwd`:/home/deep_rl_vizdoom \
       --rm\
       --entrypoint /bin/bash \
        ${image_tag}
