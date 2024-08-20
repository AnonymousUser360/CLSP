import os
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from .network.ortho_init import ortho_init_weights

from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .backbone import BackBone, BackBone_expand
from .head import ValueHead, PolicyHead, TargetUnitHead
from bigrl.core.torch_utils.data_helper import to_device

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class Model(nn.Module):
    def __init__(self, cfg={}, use_value_network=True):
        super(Model, self).__init__()
        ## 加载海拔图数据
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.model_cfg = self.whole_cfg.model
 
        self.encoder = Encoder(self.whole_cfg)
        self.backbone = BackBone(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.target_unit_head = TargetUnitHead(self.whole_cfg)
        self.use_value_network = use_value_network
        self.ortho_init = self.whole_cfg.model.get('ortho_init', False)

        

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

    def compute_logp_action(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits = self.policy_head(embedding,mask = obs["mask_info"])
        logits['target_unit']=logits_target_unit

 

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
                entropy = dist.entropy()
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
        res = {'action': action, 'action_logp': action_logp,'value': values,
               "heatmap_out_info":heatmap_out_info,'entropy':entropy}
        return res
    

    def compute_top5_logp_action(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding1 = self.backbone(embedding)
        logits_target_unit = self.target_unit_head(embedding1, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits = self.policy_head(embedding1,mask = obs["mask_info"])
        logits['target_unit']=logits_target_unit

 

        action = {}
        action_logp = {}
        top5_log_probs = []
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
                entropy = dist.entropy()/3.9889
                action[action_type] = act
                action_logp[action_type] = dist.log_prob(act)
                # 采样概率最高的5个类别
                topk_values, topk_indices = action_type_logit.topk(5, dim=1)
                # 获取每个样本的概率值
                for i in topk_indices[0]:
                    top5_log_probs.append(dist.log_prob(i).detach())
                top5_indices = topk_indices[0].numpy()
                top5_probs = torch.softmax(action_type_logit, dim=1)[:, topk_indices]
                top5_probs = np.around(top5_probs[0][0].detach().numpy(), decimals=6)


        values = {}
        if self.only_update_value:
            critic_input = embedding.detach()
        else:
            critic_input = embedding
        critic_input = torch.cat([critic_input,only_v_embedding],dim=-1)
        for k, v in self.value_networks.items():
            values[k] = v(critic_input)
        res = {'action': action, 'action_logp': action_logp,
               'value': values,"heatmap_out_info":heatmap_out_info,
               'top5_indices':top5_indices,'top5_probs':top5_probs,
               'top5_log_probs':top5_log_probs,'entropy':entropy}
        return res
    

    def compute_visdom(self, obs, **kwargs):
        embedding, only_v_embedding,heatmap_out_info,visible_enemy_embedding_no_maxpool,visible_enemy_num,visible_enemy_distance = self.encoder(obs)
        embedding = self.backbone(embedding)
        logits = self.policy_head(embedding,mask = obs["mask_info"])

        logits_target_unit = self.target_unit_head(embedding, visible_enemy_embedding_no_maxpool, visible_enemy_num) 
        logits['target_unit']=logits_target_unit

 
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
       
        # res = {'embedding': embedding}
        res = embedding
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
     
        return embedding
    
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
    
def flatten_data(data):
    if isinstance(data, dict):
        return {k: flatten_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1)
