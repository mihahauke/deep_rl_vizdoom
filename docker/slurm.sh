#!/usr/bin/env bash

sbatch -x lab-ci-2 \
        -J doom \
         --exclusive \
         -p lab-ci-student \
         docker/build_n_run_i.sh


