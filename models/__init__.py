# ------------------------------------------------------------------------
# O1O: Grouping Known Classes to Identify Unknown Objects as Odd-One-Out
# Misra Yavuz, Fatma Guney. ACCV, 2024. (https://arxiv.org/abs/2410.07514)
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


def build_model(args, mode=None):
    
    if mode == 'dn_dab_deformable_detr_o1o':
        from .dn_dab_deformable_detr_o1o import build_dab_deformable_detr
    else:
        raise NotImplementedError

    return build_dab_deformable_detr(args)

