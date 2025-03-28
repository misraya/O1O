#!/usr/bin/env bash

set -x

EXP_DIR=exps/
PY_ARGS=${@:1}
EXEMPLAR_REPLAY_DIR=o1o_sowod_exemplars
WANDB_NAME=o1o_sowod_task4_ft
DATA_DIR='data/OWOD'

mkdir -p ${EXP_DIR}/${WANDB_NAME}

python -u main_open_world_o1o.py \
    --output_dir "${EXP_DIR}/${WANDB_NAME}" --data_root ${DATA_DIR} \
    --model_type 'dn_dab_deformable_detr_o1o' \
    --train_set "${EXEMPLAR_REPLAY_DIR}/learned_owdetr_t4_ft" --test_set 'owdetr_test' \
    --dataset OWDETR --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
    --prev_num_superclass 9 --cur_num_superclass 3 --num_superclass 13 \
    --batch_size 6 --eval_every 5 --lr 0.0002 --epochs 110 \
    --use_dn --empty_weight 0.0 --sup_loss_coef 2 \
    --lr_drop 100 --lr_drop_gamma 0.1 \
    --pseudo_path "${DATA_DIR}/pseudo-boxes/OWDETR/sowod_until_t4_normal_pseudo_labels.json" \
    --pseudo_threshold 0.5 --pseudo_nms_iou 0.5 --num_unk_objects 20 \
    --resume "./exps/o1o_sowod_task4_train/checkpoint0089.pth" \
    ${PY_ARGS} > ${EXP_DIR}/${WANDB_NAME}/out.txt

