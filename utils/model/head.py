import torch.nn as nn
import torch
from typing import Optional
from .network.nn_module import fc_block2, fc_block
from .network.rnn import sequence_mask
from .network.res_block import ResFCBlock2


class ValueHead(nn.Module):
    def __init__(self, cfg):
        super(ValueHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.value
        self.embedding_dim = self.cfg.embedding_dim
        self.res_num = self.cfg.res_num
        blocks = [ResFCBlock2(in_channels=self.embedding_dim, )
                  for _ in range(self.res_num)]
        self.resnet = nn.Sequential(*blocks)
        self.output_layer = fc_block2(in_channels=self.embedding_dim,
                                      out_channels=1,
                                      norm_type=None,
                                      activation=None)

    def forward(self, x):
        x = self.resnet(x)
        x = self.output_layer(x)
        x = x.squeeze(1)
        return x


class ActionHead(nn.Module):

    def __init__(self, embedding_dim, res_num, out_dim, ):
        super(ActionHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.res_num = res_num
        self.out_dim = out_dim
        blocks = [ResFCBlock2(in_channels=self.embedding_dim, )
                  for _ in range(self.res_num)]
        self.resnet = nn.Sequential(*blocks)
        self.output_layer = fc_block2(in_channels=self.embedding_dim,
                                      out_channels=self.out_dim,
                                      norm_type=None,
                                      activation=None)

    def forward(self, x, mask = None):
        x1 = self.resnet(x)
        x2 = self.output_layer(x1)
        if mask is not None:
            x2.masked_fill_(mask.bool(), value=-1e9)
        # print(f"x1: abs mean:{torch.abs(x1).mean()},max:{x1.max()},min:{x1.min()}")
        # print(f"x2: abs mean:{torch.abs(x2).mean()},max:{x2.max()},min:{x2.min()}")
        return x2

class TargetUnitHead(nn.Module):
    def __init__(self, cfg):
        super(TargetUnitHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.policy.target_unit
        self.activation_type = self.cfg.activation
        self.key_fc = fc_block(self.cfg.entity_embedding_dim, self.cfg.key_dim, activation=None, norm_type=None)
        # self.query_mlp = nn.Sequential(*[
        #     fc_block(self.cfg.input_dim, self.cfg.key_dim, activation=self.activation_type,norm_type=None),
        #     fc_block(self.cfg.key_dim, self.cfg.key_dim, activation=None,norm_type=None),
        # ])

    def forward(
            self,
            embedding,
            player_embedding,
            player_num,
    ):
        key = self.key_fc(player_embedding)
        mask = sequence_mask(player_num, max_len=player_embedding.shape[1])
        # query = self.query_mlp(embedding)
        # backbone不过mlp了
        query = embedding
        logits = query.unsqueeze(1) * key
        logits = logits.sum(dim=2)  # b, n, -1
        logits.masked_fill_(~mask, value=-1e9)
        return logits
    




class PolicyHead(nn.Module):

    def __init__(self, cfg):
        super(PolicyHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.policy
        self.action_list = self.whole_cfg.agent.action_heads
        self.action_dim = {}
        for head in self.action_list:
            self.action_dim[head] = sum([len(self.whole_cfg.agent.actions_range[_]) \
                                        for _ in self.whole_cfg.agent.action_head_sub[head]])
            
        self.action_heads = nn.ModuleDict()
        for action_type in self.action_list:
            out_dim = self.action_dim[action_type]
            self.action_heads[action_type] = ActionHead(embedding_dim=self.cfg.embedding_dim,
                                                        res_num=self.cfg.res_num,
                                                        out_dim=out_dim)

    def forward(self,embedding,mask = None):
        res = {}
        for action_type in self.action_list:
            if mask is None:
                res[action_type] = self.action_heads[action_type](embedding,mask= mask)
            else:
                res[action_type] = self.action_heads[action_type](embedding,mask= mask[action_type])
        return res
