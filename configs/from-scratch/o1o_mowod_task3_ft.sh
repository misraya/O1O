#!/usr/bin/env bash

set -x

EXP_DIR=exps/
PY_ARGS=${@:1}
EXEMPLAR_REPLAY_DIR=o1o_mowod_exemplars
WANDB_NAME=o1o_mowod_task3_ft
DATA_DIR='data/OWOD'

mkdir -p ${EXP_DIR}/${WANDB_NAME}

python -u main_open_world_o1o.py \
    --output_dir "${EXP_DIR}/${WANDB_NAME}" --data_root ${DATA_DIR} \
    --model_type 'dn_dab_deformable_detr_o1o' \
    --train_set "${EXEMPLAR_REPLAY_DIR}/learned_owod_t3_ft" --test_set 'owod_all_task_test_nodup' \
    --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --prev_num_superclass 9 --cur_num_superclass 2 --num_superclass 13 \
    --batch_size 6 --eval_every 5 --lr 0.0002 --epochs 90 \
    --use_dn --empty_weight 0.0 --sup_loss_coef 2 \
    --lr_drop 80 --lr_drop_gamma 0.1 \
    --pseudo_path "${DATA_DIR}/pseudo-boxes/TOWOD/mowod_until_t3_normal_pseudo_labels.json" \
    --pseudo_threshold 0.5 --pseudo_nms_iou 0.5 --num_unk_objects 20 \
    --resume "./exps/o1o_mowod_task3_train/checkpoint0069.pth" \
    ${PY_ARGS} > ${EXP_DIR}/${WANDB_NAME}/out.txt

