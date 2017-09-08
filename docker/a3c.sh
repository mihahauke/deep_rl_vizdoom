#!/usr/bin/env bash

args=$@
NAME="mgr_`hostname`"
image_tag=mgr

echo 'Running docker doom on' `hostname`
echo 'Running from:': `pwd`

docker build -t ${image_tag} . 
nvidia-docker run \
       --user=`id -u`:`id -g`\
       --net=host \
       --name ${NAME} \
       -v `pwd`:/home/deep_rl_vizdoom \
       --rm\
        ${image_tag}\
	./train_a3c.py ${args}
