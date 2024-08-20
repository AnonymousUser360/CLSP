#!/user/bin/env python3
# -*- coding: utf-8 -*-


import os

import torch.nn as nn

from bigrl.core.utils.config_helper import read_config
from .network import ResFCBlock2, fc_block2

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class BackBone(nn.Module):
    def __init__(self, cfg):
        super(BackBone, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.backbone
        self.embedding_dim = self.cfg.embedding_dim
        self.res_num = self.cfg.res_num
        self.project_cfg = self.cfg.project
        self.project = fc_block2(in_channels=self.project_cfg.input_dim,
                                 out_channels=self.embedding_dim,
                                 activation='relu',
                                 norm_type='LN')
        blocks = [ResFCBlock2(in_channels=self.embedding_dim,)
                  for _ in range(self.res_num)]
        self.resnet = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.project(x)
        x = self.resnet(x)
        return x

class BackBone_expand(nn.Module):
    def __init__(self, cfg):
        super(BackBone_expand, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.backbone
        self.embedding_dim = 4096 # 4608 #4096
        self.res_num = 2
        self.project_cfg = self.cfg.project
        self.project = fc_block2(in_channels=self.project_cfg.input_dim,
                                 out_channels=self.embedding_dim,
                                 activation='relu',
                                 norm_type='LN')
        blocks = [ResFCBlock2(in_channels=self.embedding_dim,)
                  for _ in range(self.res_num)]
        self.resnet = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.project(x)
        x = self.resnet(x)
        return x
