#!/usr/bin/env bash

NAME="mgr"
image_tag="mgr"

nvidia-docker run --net=host -ti --name ${NAME} -v `pwd`:/home \
    --entrypoint /bin/bash \
    ${image_tag}