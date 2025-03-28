# ------------------------------------------------------------------------
# O1O: Grouping Known Classes to Identify Unknown Objects as Odd-One-Out
# Misra Yavuz, Fatma Guney. ACCV, 2024. (https://arxiv.org/abs/2410.07514)
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection 
# Orr Zohar, Jackson Wang, Serena Yeung
# -----------------------------------------------------------------------
# Modified from OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world_o1o import OWDetection
from engine import evaluate, train_one_epoch, get_exemplar_replay, create_ft_dataset
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    ################ Deformable DETR ################
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--lr_drop', default=35, type=int)
    # parser.add_argument('--lr_drop_epochs', default=51, type=int) 
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--masks', default=False, action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone', default='dino_resnet50', type=str, help="Name of the convolutional backbone to use")
    # Model parameters
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # Transformer
    parser.add_argument('--return_interm_layers', action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str, help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--num_results', default=300, type=int, help="Number of detection results")
    parser.add_argument('--pre_norm', action='store_true',  help="Using pre-norm in the Transformer blocks.")    
    parser.add_argument('--num_select', default=300, type=int,  help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int, help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', help="Random init the x,y of anchor boxes and freeze them.")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    # Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    # Training
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    
    ################ O1O ################
    # dataset
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int)
    parser.add_argument('--invalid_cls_logits', default=False, action='store_true', help='owod setting')
    parser.add_argument('--train_set', default='', help='training txt files')
    parser.add_argument('--test_set', default='', help='testing txt files')
    parser.add_argument('--num_classes', default=81, type=int)
    parser.add_argument('--dataset', default='OWDETR', help='defines which dataset is used. Built for: {TOWOD, OWDETR, VOC2007}')
    parser.add_argument('--data_root', default='./data/OWOD', type=str)

    # superclasses
    parser.add_argument('--num_superclass', default=1, type=int)
    parser.add_argument('--prev_num_superclass', default=1, type=int)
    parser.add_argument('--cur_num_superclass', default=1, type=int)
    parser.add_argument('--invalid_sup_logits', default='', type=str)

    # pseudo-labels
    parser.add_argument('--pseudo_path', default='', type=str)
    parser.add_argument('--alternative_pseudo_path', default='', type=str)
    parser.add_argument('--pseudo_threshold', default=0.0, type=float)
    parser.add_argument('--pseudo_nms_iou', default=0.7, type=float)
    parser.add_argument('--pseudo_backup', default='', type=str)
    parser.add_argument('--alternative_pseudo_backup', default='', type=str)
    parser.add_argument('--num_unk_objects', default=20, type=int, help="")
    parser.add_argument('--max_num_objects', default=64, type=int, help="")

    # dn dab args
    parser.add_argument('--use_dn', action="store_true", help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int, help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float, help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float, help="box noise scale to shift and scale")
    parser.add_argument('--contrastive', action="store_true", help="use contrastive training.")
    parser.add_argument('--use_mqs', action="store_true", help="use mixed query selection from DINO.")
    parser.add_argument('--use_lft', action="store_true", help="use look forward twice from DINO.")
    parser.add_argument('--amp', action='store_true', help="Train with mixed precision")
    parser.add_argument('--save_results', action='store_true', help="For eval only. Save the outputs for all images.")
    parser.add_argument('--debug', action='store_true', help="For debug only. It will perform only a few steps during trainig and val.")

    # training
    parser.add_argument('--model_type', default='prob_dab_deformable_detr', type=str)    
    parser.add_argument('--momentum', default=0.1, type=float)
    parser.add_argument('--empty_weight', default=0.1, type=float)
    parser.add_argument('--lr_drop_gamma', default=0.1, type=float)
    parser.add_argument('--num_gpus', default=0, type=int)
    parser.add_argument('--sup_loss_coef', default=1, type=float)
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    
    # exemplar replay
    parser.add_argument('--num_inst_per_class', default=50, type=int, help="number of instances per class")
    parser.add_argument('--exemplar_replay_selection', default=False, action='store_true', help='use learned exemplar selection')
    parser.add_argument('--exemplar_replay_max_length', default=1e10, type=int, help="max number of images that can be saves")
    parser.add_argument('--exemplar_replay_dir', default='', type=str, help="directory of exemplar replay txt files")
    parser.add_argument('--exemplar_replay_prev_file', default='', type=str, help="path to previous ft file")
    parser.add_argument('--exemplar_replay_cur_file', default='', type=str, help="path to current ft file")
    parser.add_argument('--exemplar_replay_random', default=False, action='store_true', help='make selection random')
    
    # eval and logging
    parser.add_argument('--eval_threshold', default=0.0, type=float)
    parser.add_argument('--wandb_name', default='', type=str)
    parser.add_argument('--wandb_project', default='O1O', type=str)

    return parser

def main(args):

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if len(args.wandb_project)>0 and utils.is_main_process():
        import wandb
        if len(args.wandb_name)>0:
            wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
        else:
            wandb.init(project=args.wandb_project,config=args)
        #wandb.config = args
    else:
        wandb=None

    print(args)

    device = torch.device(args.device)
    if args.num_gpus == 1:
        torch.cuda.set_device(device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, exemplar_selection = build_model(args, mode=args.model_type)
    model.to(device)

    model_without_ddp = model
    print(model_without_ddp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train, dataset_val = get_datasets(args) 
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, 
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, 
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, 
                                 args.batch_size, 
                                 sampler=sampler_val,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.lr_drop_gamma)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]) #, find_unused_parameters=True)
        model_without_ddp = model.module

    output_dir = Path(args.output_dir)

    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load( args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)

        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, dataset_val, device, args.output_dir, args=args,
                                            eval_mode=True, eval_threshold=args.eval_threshold)                
            if wandb is not None and utils.is_main_process():
                wandb.log({str(key): val for key, val in test_stats["metrics"].items()})        

        if args.exemplar_replay_selection:
            with torch.no_grad():
                model.eval()        
                image_sorted_scores = get_exemplar_replay(model, exemplar_selection, device, data_loader_train, args)
                create_ft_dataset(args, image_sorted_scores)        
        return
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.gamma = args.lr_drop_gamma

            args.start_epoch = checkpoint['epoch'] + 1
            # if checkpoint['epoch'] > args.lr_drop_epochs:
            #     lr_scheduler.step(checkpoint['epoch'] - args.lr_drop_epochs)
            if checkpoint['epoch'] > args.lr_drop:
                lr_scheduler.step(checkpoint['epoch'] - args.lr_drop)
            # elif checkpoint['epoch'] == args.lr_drop_epochs:
            elif checkpoint['epoch'] == args.lr_drop:
                lr_scheduler.step(args.lr_drop)

            log_stats = {'train_lr': lr_scheduler.get_last_lr()[0], 'epoch': args.start_epoch}

            if wandb is not None and utils.is_main_process():
                wandb.log(log_stats)
            
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, dataset_val, device, args.output_dir, args=args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            
    print(f'Start training from epoch {args.start_epoch} to {args.epochs}')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # if epoch == args.lr_drop_epochs:
        if epoch == args.lr_drop:
            lr_scheduler.step(args.lr_drop)
        elif epoch > args.lr_drop:
            lr_scheduler.step()

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, lr_scheduler=lr_scheduler, args=args, logger=None, wandb=wandb, 
        )
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.eval_every == 0 or epoch == 0 or epoch == 1 or (args.epochs-epoch)<1:
                test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, 
                                                  data_loader_val, dataset_val, device, args.output_dir, args=args)
                # checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                if wandb is not None and utils.is_main_process():
                    test_stats["metrics"]['epoch']=epoch
                    wandb.log({str(key): val for key, val in test_stats["metrics"].items()})                
            else:
                 test_stats = {}
                    
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if wandb is not None and utils.is_main_process():
            wandb.log(log_stats)
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.dataset in ['owod', 'owdetr'] and (epoch+1) % args.eval_every == 0 and epoch > 0:
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
                            
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.exemplar_replay_selection:
        with torch.no_grad():
            model.eval()        
            image_sorted_scores = get_exemplar_replay(model, exemplar_selection, device, data_loader_train, args)
            create_ft_dataset(args, image_sorted_scores)        
    

def get_datasets(args):
    print(args.train_set)
    print(args.test_set)

    dataset_train = OWDetection(args, 
                    args.data_root, 
                    image_set=args.train_set, 
                    dataset = args.dataset, 
                    pseudo_path=args.pseudo_path,
                    transforms=make_coco_transforms(args.train_set))
    dataset_val = OWDetection(args, 
                    args.data_root, 
                    image_set=args.test_set, 
                    dataset = args.dataset, 
                    transforms=make_coco_transforms(args.test_set))

    print(dataset_train)
    print(dataset_val)

    dataset_val.superclass_names = dataset_train.superclass_names
    dataset_val.superclass_indices = dataset_train.superclass_indices

    return dataset_train, dataset_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)