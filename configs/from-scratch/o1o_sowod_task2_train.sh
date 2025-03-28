#!/usr/bin/env bash

set -x

EXP_DIR=exps/
PY_ARGS=${@:1}
EXEMPLAR_REPLAY_DIR=o1o_sowod_exemplars
WANDB_NAME=o1o_sowod_task2_train
DATA_DIR='data/OWOD'

mkdir -p ${EXP_DIR}/${WANDB_NAME}

python -u main_open_world_o1o.py \
    --output_dir "${EXP_DIR}/${WANDB_NAME}" --data_root ${DATA_DIR} \
    --model_type 'dn_dab_deformable_detr_o1o' \
    --train_set 'owdetr_t2_train' --test_set 'owdetr_test' \
    --dataset OWDETR --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 \
    --prev_num_superclass 3 --cur_num_superclass 4 --num_superclass 13 \
    --batch_size 6 --eval_every 1 --lr 0.00002 --epochs 30 \
    --use_dn --empty_weight 0.0 --sup_loss_coef 2 \
    --lr_drop 40 --lr_drop_gamma 0.1 \
    --pseudo_path "${DATA_DIR}/pseudo-boxes/OWDETR/sowod_until_t2_normal_pseudo_labels.json" \
    --pseudo_threshold 0.5 --pseudo_nms_iou 0.5 --num_unk_objects 20 \
    --resume "./exps/o1o_sowod_task1_train/checkpoint0019.pth" \
    --exemplar_replay_selection --exemplar_replay_dir ${EXEMPLAR_REPLAY_DIR}  --exemplar_replay_max_length 1679\
    --exemplar_replay_prev_file "learned_owdetr_t1_ft.txt" --exemplar_replay_cur_file "learned_owdetr_t2_ft.txt"\
    ${PY_ARGS} > ${EXP_DIR}/${WANDB_NAME}/out.txt

