#!/usr/bin/env bash

GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task1_train.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task1_eval.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task2_train.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task2_ft.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task2_eval.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task3_train.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task3_ft.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task3_eval.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task4_train.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task4_ft.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/from-scratch/o1o_sowod_task4_eval.sh
