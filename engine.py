# ------------------------------------------------------------------------
# O1O: Grouping Known Classes to Identify Unknown Objects as Odd-One-Out
# Misra Yavuz, Fatma Guney. ACCV, 2024. (https://arxiv.org/abs/2410.07514)
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DN-DETR: Accelerate DETR Training by Introducing Query DeNoising
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import numpy as np
from typing import Iterable
import torch
import util.misc as utils
from datasets.open_world_eval import OWEvaluator

def slprint(x, name='x'):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        print(f'{name}.shape:', x.shape)
    elif isinstance(x, (tuple, list)):
        print('type x:', type(x))
        for i in range(min(10, len(x))):
            slprint(x[i], f'{name}[{i}]')
    elif isinstance(x, dict):
        for k,v in x.items():
            slprint(v, f'{name}[{k}]')
    else:
        print(f'{name}.type:', type(x))


def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k,v in item.items()}
    else:
        raise NotImplementedError("Call Shilong if you use other containers! type: {}".format(type(item)))

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None, wandb=None,
                    ):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                dn_args=(targets, args.scalar, args.label_noise_scale, args.box_noise_scale, args.num_patterns)
                if args.contrastive is not False:
                    dn_args += (args.contrastive,)

                outputs, mask_dict = model(samples, dn_args=dn_args)
                loss_dict = criterion(outputs, targets, mask_dict, epoch=epoch)
            else:
                outputs = model(samples)
                loss_dict = criterion(outputs, targets, epoch=epoch)

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()


        if wandb is not None and utils.is_main_process():
            wandb.log({"total_loss":loss_value})
            wandb.log(loss_dict_reduced_scaled)
            wandb.log(loss_dict_reduced_unscaled)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None,
             eval_mode=False, eval_threshold=0.0):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args)

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # import ipdb; ipdb.set_trace()
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:

                if hasattr(args, 'cascade_attn') and args.cascade_attn:
                    outputs, _ = model(samples, dn_args=(targets,args.num_patterns))
                else:
                    outputs, _ = model(samples, dn_args=args.num_patterns)

            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, eval_mode=eval_mode, eval_threshold=eval_threshold)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import ipdb; ipdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        
        if args.save_results:
            """
            saving results of eval.
            """
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()[1]
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats['metrics']=res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # import ipdb; ipdb.set_trace()
    return stats, coco_evaluator

    
@torch.no_grad()
def get_exemplar_replay(model, exemplar_selection, device, data_loader, args):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[ExempReplay]'
    print_freq = 10

    image_sorted_scores_reduced={}
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, _ = model(samples, dn_args=args.num_patterns)
            else:
                outputs = model(samples)

        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
            
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        
    print(f'found a total of {len(image_sorted_scores_reduced.keys())} images')
    return image_sorted_scores_reduced


def create_ft_dataset(args, image_sorted_scores):
    print(f'found a total of {len(image_sorted_scores.keys())} images')

    tmp_dir=args.data_root +'/ImageSets/'+args.dataset+"/"+str(args.exemplar_replay_dir)+"/"

    class_sorted_scores={}
    imgs_per_class={}
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS):
        class_sorted_scores[str(i)]=[]
        imgs_per_class[str(i)]=[]

    for k,v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            if v['labels'][j] >= args.PREV_INTRODUCED_CLS and v['labels'][j] < args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS:
                class_sorted_scores[str(v['labels'][j])].append(v['scores'][j])

    class_threshold={}
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS):
        tmp=np.array(class_sorted_scores[str(i)])
        tmp.sort()
        tmp = torch.Tensor(tmp)
        if len(tmp)>args.num_inst_per_class and not args.exemplar_replay_random:
            max_val = tmp[-args.num_inst_per_class//2]
            min_val = tmp[args.num_inst_per_class//2]
        else:
            if args.exemplar_replay_random:
                print('using random exemplar selection')
            else:
                print(f'only found {len(tmp)} imgs in class {i}')
            max_val = tmp.min()
            min_val = tmp.max()
            
        class_threshold[str(i)]=(min_val, max_val)

    save_imgs = []    
    for k,v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            if v['labels'][j] >= args.PREV_INTRODUCED_CLS and v['labels'][j] < args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS:                
                label = str(v['labels'][j])
                if (v['scores'][j] <= class_threshold[label][0].numpy() or v['scores'][j] >= class_threshold[label][1].numpy()) and (len(imgs_per_class[label])<=args.num_inst_per_class+2):
                    img = str(k)[4:]
                    if len(img)==10:
                        imgs_per_class[label].append(img[:4]+'_'+img[4:])
                        save_imgs.append(img[:4]+'_'+img[4:])
                    else:
                        save_imgs.append(img)
                        imgs_per_class[label].append(img)

                        
    print(f'found {len(np.unique(save_imgs))} images in run')

    if len(args.exemplar_replay_prev_file)>0:
        previous_ft = open(tmp_dir+args.exemplar_replay_prev_file,'r').read().splitlines()
        save_imgs+=previous_ft
        
    save_imgs=np.unique(save_imgs)
    np.random.shuffle(save_imgs)
    if len(save_imgs)> args.exemplar_replay_max_length:
        save_imgs=save_imgs[:args.exemplar_replay_max_length]
    
    os.makedirs(tmp_dir, exist_ok=True)
    print('saving to:', tmp_dir+args.exemplar_replay_cur_file)
    with open(tmp_dir+args.exemplar_replay_cur_file, 'w') as f:
        for line in save_imgs:
            f.write(line)
            f.write('\n')

