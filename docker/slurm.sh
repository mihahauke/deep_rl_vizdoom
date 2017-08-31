#!/usr/bin/env bash

sbatch  -J doom \
        --exclusive \
        -p lab-ci-student \
        docker/build_n_run_i.sh $@


