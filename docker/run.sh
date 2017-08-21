#!/usr/bin/env bash

NAME="mgr"
image_tag="mgr"

nvidia-docker run --net=host -ti --rm --name ${NAME} -v `pwd`:/home \
${image_tag}
