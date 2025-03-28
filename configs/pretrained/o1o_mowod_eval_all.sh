#!/usr/bin/env bash

GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/pretrained/o1o_mowod_task1_eval.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/pretrained/o1o_mowod_task2_eval.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/pretrained/o1o_mowod_task3_eval.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/pretrained/o1o_mowod_task4_eval.sh
