import os
from IPython import embed
from flask import config
import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
import config as CFG
from utils.model.encoder import Encoder
from utils.model.backbone import BackBone,BackBone_expand 
from utils.model.model import Model_expand,Model
from utils_clip import read_config,deep_merge_dicts

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.image_model_name, pretrained=CFG.image_pretrained, trainable=CFG.image_trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

class StateEncoder(nn.Module):
    """
    Encode states to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.state_pretrained, trainable=CFG.state_trainable
    ):
        super().__init__()
 
        cfg = read_config('./utils/user_config_replay.yaml')
        default_cfg = read_config('./utils/model/default_model_config_gaussian.yaml')
        self.whole_cfg = deep_merge_dicts(default_cfg, cfg)  
 
        if CFG.state_embedding ==  4096:
            self.backbone = Model_expand(self.whole_cfg)
            print(f"!!!!!!!!!!!!!!!!Using Model expand")
        elif CFG.state_embedding ==  256:
            self.backbone = Model(self.whole_cfg)
            print(f"Using Model !!!!!!!!!!!!!!!!!!!!!!")
        
        if pretrained:          
            pretrained_model = torch.load(f'./models/{CFG.state_encoder_pretrained_name}')                      
            self.backbone.load_state_dict(pretrained_model)
        else:
            self.init_params()
 
 
        for p in self.backbone.parameters():
            p.requires_grad = trainable
 
        

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        embedding = self.backbone.get_backbone_embedding(x) 
        return embedding


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.text_pretrained, trainable=CFG.text_trainable):
        super().__init__()
 
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            config=DistilBertConfig()
            config.dim=CFG.text_embedding
            config.hidden_dim =CFG.text_embedding*4
            self.model = DistilBertModel(config=config)
            self.init_params()
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

 

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids , attention_mask=attention_mask )
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
 



class TextEncoder_AddFC(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.text_pretrained, trainable=CFG.text_trainable):
        super().__init__()

        self.heads =  nn.Sequential(
            nn.Linear(CFG.text_embedding, CFG.text_hidden_size),
            nn.ReLU(), 
            nn.Linear(CFG.text_hidden_size, CFG.text_expand_embedding)
        ) 
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            config=DistilBertConfig()
            config.dim=CFG.text_embedding
            config.hidden_dim =CFG.text_embedding*4
            config.RFF_dim  = 512
            config.RFF_sigma  = 1
            self.model = DistilBertModel(config=config)
            self.init_params()

        self.initialize_heads_weights()
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

 
    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight)

    def initialize_heads_weights(self):
        for module in self.heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias) 
                                                  
    # def forward(self, input_ids, attention_mask,segments_length_pad=None,key_values_unroll_pad=None,key_values_segement_lens_pad=None,segment_types=None):
    def forward(self, input_ids, attention_mask):
        # if segments_length_pad is not None:
        #     output = self.model(input_ids=input_ids , attention_mask=attention_mask,segments_length_pad=segments_length_pad,key_values_unroll_pad=key_values_unroll_pad,key_values_segement_lens_pad=key_values_segement_lens_pad,segment_types=segment_types )
        # else:
        output = self.model(input_ids=input_ids , attention_mask=attention_mask )
        last_hidden_state = output.last_hidden_state
        src_output = last_hidden_state[:, self.target_token_idx, :]
        resized_output = self.heads(src_output)
        return resized_output


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

