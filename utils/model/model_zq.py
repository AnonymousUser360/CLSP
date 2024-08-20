import os
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from .network.ortho_init import ortho_init_weights

from ..config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .backbone import BackBone, BackBone_expand
from .backbone_for_llm import BackBoneTokens
from .head import ValueHead, PolicyHead, TargetUnitHead
# from bigrl.core.torch_utils.data_helper import to_device

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class MultiLinearJupiter(nn.Module):
    '''
    这是为了实现将单个embedding转化为多个token，速度贼快
    x: (batch_size,input_size)
    y: (batch_size,num_heads,output_size)
    '''

    def __init__(self, input_size, output_size, num_heads, bias=False):
        super(MultiLinearJupiter, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_heads, input_size, output_size))
        if (bias):
            self.bias = nn.Parameter(torch.randn(num_heads, output_size))
        else:
            self.bias = None

    def forward(self, x):
        y = torch.matmul(x, self.weights).transpose(1, 0)
        if (self.bias is not None):
            y = y + self.bias
        return y


class MultiMLPJupiter(nn.Module):
    '''
    这是为了实现将单个embedding转化为多个token，速度贼快
    x: (batch_size,input_size)
    y: (batch_size,num_heads,output_size)
    '''


    def __init__(self, input_size, output_size, num_heads, bias=False, bottle_neck_size=64):
        super(MultiMLPJupiter, self).__init__()
        self.multi_linear = MultiLinearJupiter(input_size=input_size, output_size=bottle_neck_size, num_heads=num_heads,
                                               bias=bias)
        self.num_heads = num_heads
        self.bottle_neck_size = bottle_neck_size
        self.linear_mat = nn.Parameter(torch.randn(num_heads, bottle_neck_size, bottle_neck_size))
        if (bias):
            self.bias = nn.Parameter(torch.randn(num_heads, bottle_neck_size))
        else:
            self.bias = None
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.shared_embedding = nn.Parameter(torch.randn(bottle_neck_size, output_size))
        self.init_params()

    def forward(self, x):
        # B*H*64
        y = self.multi_linear(x)
        y = self.gelu1(y)
        y = self.fast_complex_matmul(y, self.linear_mat)
        # y=self.slow_complex_matmul(y, self.linear_mat)
        y = self.gelu2(y)
        y = torch.matmul(y, self.shared_embedding)
        return y

    def fast_complex_matmul(self, input, paras):
        batch_size, num_heads, feature_size = input.shape

        # 重排 input 形状以适应批量矩阵乘法
        # input 初始形状为 (batch_size, num_heads, feature_size)
        input = input.transpose(1, 0)  # 更改为 (num_heads, batch_size, feature_size)

        # paras 形状为 (num_heads, feature_size, feature_size)

        # 执行批量矩阵乘法
        result = torch.matmul(input, paras)  # result 形状将为 (num_heads, batch_size, feature_size)

        # 转置回 (batch_size, num_heads, feature_size)
        result = result.transpose(1, 0)

        return result

    def slow_complex_matmul(self, input, paras):
        result = []
        for i in range(self.num_heads):
            result.append(torch.matmul(input[:, i, :], paras[i]))
        return torch.stack(result).transpose(1, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)

class Model(nn.Module):
    def __init__(self, cfg={}, use_value_network=True):
        super(Model, self).__init__()
        ## 加载海拔图数据
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.model_cfg = self.whole_cfg.model
        # heat_map_data = np.load(self.whole_cfg.env.heat_map.heat_map_path)
        # print("loading heat map, please wait!!!!!!!!!!!!!!!")
        # heat_map_data = torch.tensor(heat_map_data,dtype=torch.float)
        # heat_map_data = to_device(heat_map_data, device)
        self.encoder = Encoder(self.whole_cfg)
        self.backbone = BackBone(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.target_unit_head = TargetUnitHead(self.whole_cfg)
        self.use_value_network = use_value_network
        self.ortho_init = self.whole_cfg.model.get('ortho_init', False)
        self.init_params()

        if use_value_network:
            self.only_update_value = False
            self.value_networks = nn.ModuleDict()
            self.value_head_init_gains = self.whole_cfg.model.get('value_head_init_gains', {})
            for k in self.whole_cfg.agent.enable_baselines:
                self.value_networks[k] = ValueHead(self.whole_cfg)
                if self.ortho_init:
                    ortho_init_weights(self.value_networks[k],gain=np.sqrt(2))
                    ortho_init_weights(self.value_networks[k].output_layer,gain=self.value_head_init_gains.get(k,1))

    def forward(self, obs):
        embedding, player_embedding, player_num = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits = self.policy_head(embedding,player_embedding,player_num)
        return logits

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)

    def compute_logp_action(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits = self.policy_head(embedding,mask = obs["mask_info"])
        logits['target_unit']=logits_target_unit

        #  # TODO 由最近的敌人逐渐过渡到attention选择的敌人，通过在最近敌人那个值上加上一个数值的方式
        # if visible_enemy_num>0:
        #     logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num)               
        #     logits_target_unit.div_(self.whole_cfg['model']['policy']['target_unit'].temperature)        
        #     prob_target_unit = torch.nn.functional.softmax(logits_target_unit, dim=1)
        #     closest_enemy_idx = visible_enemy_distance[0,:visible_enemy_num].argmin(dim=-1)
        #     prob_target_unit[0][closest_enemy_idx] += self.whole_cfg['agent'].act_attention_shift
        #     prob_target_unit = torch.nn.functional.softmax(prob_target_unit, dim=1)
        #     target_unit = torch.multinomial(prob_target_unit, num_samples=1, replacement=True)[:, 0]
        # else:
        #     target_unit = None

        action = {}
        action_logp = {}
        for action_type, action_type_logit in logits.items():
            if action_type == "target_unit":
                action_type_logit.div_(self.whole_cfg['model']['policy']['target_unit'].temperature)
                prob_target_unit = torch.nn.functional.softmax(action_type_logit, dim=-1)
                # TODO 概率加了一个偏移
                prob_target_unit[0][0] += obs["visible_enemy_item_info"]["act_attention_shift"][0]
                prob_target_unit = prob_target_unit/(1 + obs["visible_enemy_item_info"]["act_attention_shift"]) 
                target_unit = torch.multinomial(prob_target_unit, num_samples=1, replacement=True)[:, 0]
                action[action_type] = target_unit
                action_logp[action_type] = torch.log(prob_target_unit[0][target_unit])
            else:
                dist = torch.distributions.Categorical(logits=action_type_logit)
                act = dist.sample()
                action[action_type] = act
                action_logp[action_type] = dist.log_prob(act)
        values = {}
        if self.only_update_value:
            critic_input = embedding.detach()
        else:
            critic_input = embedding
        critic_input = torch.cat([critic_input,only_v_embedding],dim=-1)
        for k, v in self.value_networks.items():
            values[k] = v(critic_input)
        res = {'action': action, 'action_logp': action_logp,'value': values,"heatmap_out_info":heatmap_out_info}
        return res
    def compute_visdom(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits = self.policy_head(embedding,mask = obs["mask_info"])

        logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits['target_unit']=logits_target_unit

        # # TODO 由最近的敌人逐渐过渡到attention选择的敌人，通过在最近敌人那个值上加上一个数值的方式
        # if visible_enemy_num>0:
        #     logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num)               
        #     logits_target_unit.div_(self.whole_cfg['model']['policy']['target_unit'].temperature)        
        #     prob_target_unit = torch.nn.functional.softmax(logits_target_unit, dim=1)
        #     closest_enemy_idx = visible_enemy_distance[0,:visible_enemy_num].argmin(dim=-1)
        #     prob_target_unit[0][closest_enemy_idx] += self.whole_cfg['agent'].act_attention_shift
        #     prob_target_unit = torch.nn.functional.softmax(prob_target_unit, dim=1)
        #     target_unit = torch.multinomial(prob_target_unit, num_samples=1, replacement=True)[:, 0]
        # else:
        #     target_unit = None
        action = {}
        action_logp = {}
        for action_type, action_type_logit in logits.items():
            if action_type == "target_unit":
                action_type_logit.div_(self.whole_cfg['model']['policy']['target_unit'].temperature)
                prob_target_unit = torch.nn.functional.softmax(action_type_logit, dim=-1)
                # TODO 概率加了一个偏移
                prob_target_unit[0][0] += obs["visible_enemy_item_info"]["act_attention_shift"][0]
                prob_target_unit = prob_target_unit/(1 + obs["visible_enemy_item_info"]["act_attention_shift"]) 
                target_unit = torch.multinomial(prob_target_unit, num_samples=1, replacement=True)[:, 0]
                action[action_type] = target_unit
                action_logp[action_type] = torch.log(prob_target_unit[0][target_unit])
            else:
                dist = torch.distributions.Categorical(logits=action_type_logit)
                act = dist.sample()
                action[action_type] = act
                action_logp[action_type] = dist.log_prob(act)
        values = {}
        if self.only_update_value:
            critic_input = embedding.detach()
        else:
            critic_input = embedding
        critic_input = torch.cat([critic_input,only_v_embedding],dim=-1)
        for k, v in self.value_networks.items():
            values[k] = v(critic_input)
        res = {'action': action, 'action_logp': action_logp,'value': values,"heatmap_out_info":heatmap_out_info,}
        return res

    def get_backbone_embedding(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
       
        res = {'embedding': embedding}
        return res
    
    def rl_train(self, inputs: dict, **kwargs) -> Dict[str, Any]:
        obs = inputs['obs']
        batch_size = inputs['done'].shape[1]
        unroll_len = inputs['done'].shape[0]
        

        flatten_obs = flatten_data(obs)
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(flatten_obs)
        embedding = self.backbone(embedding)
        logits = self.policy_head(embedding[:-batch_size],mask={k:v[:-batch_size] for k,v in flatten_obs["mask_info"].items()})
        # TODO 记得修改过来
        logits_target_unit = self.target_unit_head(embedding[:-batch_size], visible_enemy_embedding_no_maxpool[:-batch_size], visible_enemy_num[:-batch_size]) 

        logits['target_unit']=logits_target_unit

        values = {}
        if self.only_update_value:
            critic_input = embedding.detach()
        else:
            critic_input = embedding
        critic_input = torch.cat([critic_input,only_v_embedding],dim=-1)
        for k, v in self.value_networks.items():
            values[k] = v(critic_input)
        model_res = {'logit': logits,
                     'action': inputs['action'],
                     'action_logp': inputs['action_logp'],
                     "advantage":inputs['advantage'],
                        "return":inputs['return'],
                     'value': values,
                     'reward': inputs['reward'],
                     'done': inputs['done'],
                     'flatten_obs': flatten_obs,
                     }
        return model_res

    # used in rl teacher model
    def teacher_forward(self, obs):

        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits = self.policy_head(embedding,mask = obs["mask_info"])
        logits['target_unit']=logits_target_unit

        return logits

class Model_expand(nn.Module):
    def __init__(self, cfg={}, use_value_network=True):
        super(Model_expand, self).__init__()
        ## 加载海拔图数据
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.model_cfg = self.whole_cfg.model
        # heat_map_data = np.load(self.whole_cfg.env.heat_map.heat_map_path)
        # print("loading heat map, please wait!!!!!!!!!!!!!!!")
        # heat_map_data = torch.tensor(heat_map_data,dtype=torch.float)
        # heat_map_data = to_device(heat_map_data, device)
        self.encoder = Encoder(self.whole_cfg)
        self.backbone = BackBone_expand(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.target_unit_head = TargetUnitHead(self.whole_cfg)
        self.use_value_network = use_value_network
        self.ortho_init = self.whole_cfg.model.get('ortho_init', False)
        self.init_params()

        if use_value_network:
            self.only_update_value = False
            self.value_networks = nn.ModuleDict()
            self.value_head_init_gains = self.whole_cfg.model.get('value_head_init_gains', {})
            for k in self.whole_cfg.agent.enable_baselines:
                self.value_networks[k] = ValueHead(self.whole_cfg)
                if self.ortho_init:
                    ortho_init_weights(self.value_networks[k],gain=np.sqrt(2))
                    ortho_init_weights(self.value_networks[k].output_layer,gain=self.value_head_init_gains.get(k,1))

    def forward(self, obs):
        embedding, player_embedding, player_num = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits = self.policy_head(embedding,player_embedding,player_num)
        return logits

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)

    def compute_logp_action(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits = self.policy_head(embedding,mask = obs["mask_info"])
        logits['target_unit']=logits_target_unit

        #  # TODO 由最近的敌人逐渐过渡到attention选择的敌人，通过在最近敌人那个值上加上一个数值的方式
        # if visible_enemy_num>0:
        #     logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num)               
        #     logits_target_unit.div_(self.whole_cfg['model']['policy']['target_unit'].temperature)        
        #     prob_target_unit = torch.nn.functional.softmax(logits_target_unit, dim=1)
        #     closest_enemy_idx = visible_enemy_distance[0,:visible_enemy_num].argmin(dim=-1)
        #     prob_target_unit[0][closest_enemy_idx] += self.whole_cfg['agent'].act_attention_shift
        #     prob_target_unit = torch.nn.functional.softmax(prob_target_unit, dim=1)
        #     target_unit = torch.multinomial(prob_target_unit, num_samples=1, replacement=True)[:, 0]
        # else:
        #     target_unit = None

        action = {}
        action_logp = {}
        for action_type, action_type_logit in logits.items():
            if action_type == "target_unit":
                action_type_logit.div_(self.whole_cfg['model']['policy']['target_unit'].temperature)
                prob_target_unit = torch.nn.functional.softmax(action_type_logit, dim=-1)
                # TODO 概率加了一个偏移
                prob_target_unit[0][0] += obs["visible_enemy_item_info"]["act_attention_shift"][0]
                prob_target_unit = prob_target_unit/(1 + obs["visible_enemy_item_info"]["act_attention_shift"]) 
                target_unit = torch.multinomial(prob_target_unit, num_samples=1, replacement=True)[:, 0]
                action[action_type] = target_unit
                action_logp[action_type] = torch.log(prob_target_unit[0][target_unit])
            else:
                dist = torch.distributions.Categorical(logits=action_type_logit)
                act = dist.sample()
                action[action_type] = act
                action_logp[action_type] = dist.log_prob(act)
        values = {}
        if self.only_update_value:
            critic_input = embedding.detach()
        else:
            critic_input = embedding
        critic_input = torch.cat([critic_input,only_v_embedding],dim=-1)
        for k, v in self.value_networks.items():
            values[k] = v(critic_input)
        res = {'action': action, 'action_logp': action_logp,'value': values,"heatmap_out_info":heatmap_out_info}
        return res
    def compute_visdom(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits = self.policy_head(embedding,mask = obs["mask_info"])

        logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits['target_unit']=logits_target_unit

        # # TODO 由最近的敌人逐渐过渡到attention选择的敌人，通过在最近敌人那个值上加上一个数值的方式
        # if visible_enemy_num>0:
        #     logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num)               
        #     logits_target_unit.div_(self.whole_cfg['model']['policy']['target_unit'].temperature)        
        #     prob_target_unit = torch.nn.functional.softmax(logits_target_unit, dim=1)
        #     closest_enemy_idx = visible_enemy_distance[0,:visible_enemy_num].argmin(dim=-1)
        #     prob_target_unit[0][closest_enemy_idx] += self.whole_cfg['agent'].act_attention_shift
        #     prob_target_unit = torch.nn.functional.softmax(prob_target_unit, dim=1)
        #     target_unit = torch.multinomial(prob_target_unit, num_samples=1, replacement=True)[:, 0]
        # else:
        #     target_unit = None
        action = {}
        action_logp = {}
        for action_type, action_type_logit in logits.items():
            if action_type == "target_unit":
                action_type_logit.div_(self.whole_cfg['model']['policy']['target_unit'].temperature)
                prob_target_unit = torch.nn.functional.softmax(action_type_logit, dim=-1)
                # TODO 概率加了一个偏移
                prob_target_unit[0][0] += obs["visible_enemy_item_info"]["act_attention_shift"][0]
                prob_target_unit = prob_target_unit/(1 + obs["visible_enemy_item_info"]["act_attention_shift"]) 
                target_unit = torch.multinomial(prob_target_unit, num_samples=1, replacement=True)[:, 0]
                action[action_type] = target_unit
                action_logp[action_type] = torch.log(prob_target_unit[0][target_unit])
            else:
                dist = torch.distributions.Categorical(logits=action_type_logit)
                act = dist.sample()
                action[action_type] = act
                action_logp[action_type] = dist.log_prob(act)
        values = {}
        if self.only_update_value:
            critic_input = embedding.detach()
        else:
            critic_input = embedding
        critic_input = torch.cat([critic_input,only_v_embedding],dim=-1)
        for k, v in self.value_networks.items():
            values[k] = v(critic_input)
        res = {'action': action, 'action_logp': action_logp,'value': values,"heatmap_out_info":heatmap_out_info,}
        return res

    def get_backbone_embedding(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
       
        res = {'embedding': embedding}
        return res
    
    def rl_train(self, inputs: dict, **kwargs) -> Dict[str, Any]:
        obs = inputs['obs']
        batch_size = inputs['done'].shape[1]
        unroll_len = inputs['done'].shape[0]
        

        flatten_obs = flatten_data(obs)
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(flatten_obs)
        embedding = self.backbone(embedding)
        logits = self.policy_head(embedding[:-batch_size],mask={k:v[:-batch_size] for k,v in flatten_obs["mask_info"].items()})
        # TODO 记得修改过来
        logits_target_unit = self.target_unit_head(embedding[:-batch_size], visible_enemy_embedding_no_maxpool[:-batch_size], visible_enemy_num[:-batch_size]) 

        logits['target_unit']=logits_target_unit

        values = {}
        if self.only_update_value:
            critic_input = embedding.detach()
        else:
            critic_input = embedding
        critic_input = torch.cat([critic_input,only_v_embedding],dim=-1)
        for k, v in self.value_networks.items():
            values[k] = v(critic_input)
        model_res = {'logit': logits,
                     'action': inputs['action'],
                     'action_logp': inputs['action_logp'],
                     "advantage":inputs['advantage'],
                        "return":inputs['return'],
                     'value': values,
                     'reward': inputs['reward'],
                     'done': inputs['done'],
                     'flatten_obs': flatten_obs,
                     }
        return model_res

    # used in rl teacher model
    def teacher_forward(self, obs):

        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits = self.policy_head(embedding,mask = obs["mask_info"])
        logits['target_unit']=logits_target_unit

        return logits

class Model_Plus(nn.Module):
    '''
    对比原rl的backbone，修改输出为多个token
    '''
    def __init__(self, cfg={}, use_value_network=True):
        super(Model_Plus, self).__init__()
        ## 加载海拔图数据
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.model_cfg = self.whole_cfg.model
        # heat_map_data = np.load(self.whole_cfg.env.heat_map.heat_map_path)
        # print("loading heat map, please wait!!!!!!!!!!!!!!!")
        # heat_map_data = torch.tensor(heat_map_data,dtype=torch.float)
        # heat_map_data = to_device(heat_map_data, device)
        self.encoder = Encoder(self.whole_cfg)
        self.backboneT = BackBoneTokens(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.target_unit_head = TargetUnitHead(self.whole_cfg)
        self.use_value_network = use_value_network
        self.ortho_init = self.whole_cfg.model.get('ortho_init', False)
        self.init_params()

        if use_value_network:
            self.only_update_value = False
            self.value_networks = nn.ModuleDict()
            self.value_head_init_gains = self.whole_cfg.model.get('value_head_init_gains', {})
            for k in self.whole_cfg.agent.enable_baselines:
                self.value_networks[k] = ValueHead(self.whole_cfg)
                if self.ortho_init:
                    ortho_init_weights(self.value_networks[k],gain=np.sqrt(2))
                    ortho_init_weights(self.value_networks[k].output_layer,gain=self.value_head_init_gains.get(k,1))

    def forward(self, obs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backboneT(embedding)
        return embedding

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)

    def get_backbone_embedding(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backboneT(embedding)
       
        res = {'embedding': embedding}
        return res


def flatten_data(data):
    if isinstance(data, dict):
        return {k: flatten_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1)
