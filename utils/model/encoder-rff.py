import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .network.encoder import SignBinaryEncoder, BinaryEncoder, OnehotEncoder, TimeEncoder, UnsqueezeEncoder,FrequencyEncoder,GaussianEncoder
from .network.nn_module import fc_block, MLP
from .network.res_block import ResBlock2, conv2d_block2,ResBlock, conv2d_block
from .network.transformer import Transformer
from .network.rnn import sequence_mask


class ScalarEncoder(nn.Module):
    def __init__(self, cfg,name=str):
        super(ScalarEncoder, self).__init__()
        self.whole_cfg = cfg
        self.name = name
        self.cfg = self.whole_cfg.model[self.name + '_encoder']
        self.encode_modules = nn.ModuleDict()
        for k, item in self.cfg.modules.items():
            if item['arc'] == 'time':
                self.encode_modules[k] = TimeEncoder(embedding_dim=item['embedding_dim'])
            elif item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'frequency':
                self.encode_modules[k] = FrequencyEncoder(embedding_dim=item['embedding_dim'], )
            elif item['arc'] == 'gaussian':
                self.encode_modules[k] = GaussianEncoder(embedding_dim=item['embedding_dim'], )
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'unsqueeze':
                self.encode_modules[k] = UnsqueezeEncoder(norm_value=item['norm_value'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError

        self.layers = MLP(in_channels=self.cfg.input_dim, hidden_channels=self.cfg.hidden_dim,
                          out_channels=self.cfg.output_dim,
                          layer_num=self.cfg.layer_num,
                          layer_fn=fc_block,
                          activation=self.cfg.activation,
                          norm_type=self.cfg.norm_type,
                          use_dropout=False
                          )

    def forward(self, x: Dict[str, Tensor]):
        embeddings = []
        for key, item in self.cfg.modules.items():
            assert key in x, key
            embeddings.append(self.encode_modules[key](x[key]))

        out = torch.cat(embeddings, dim=-1)
        out = self.layers(out)
        return out
    
    
class HistoryPositionEncoder(nn.Module):
    def __init__(self, cfg):
        super(HistoryPositionEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model["history_position_encoder"]
        self.layers = MLP(in_channels=self.cfg.input_dim, hidden_channels=self.cfg.hidden_dim,
                          out_channels=self.cfg.output_dim,
                          layer_num=self.cfg.layer_num,
                          layer_fn=fc_block,
                          activation=self.cfg.activation,
                          norm_type=self.cfg.norm_type,
                          use_dropout=False
                          )
    def forward(self, x: Tensor):
        out = self.layers(x)
        return out
        

class TransformerEncoder(nn.Module):
    def __init__(self, cfg,name:str,example_key:str):
        super(TransformerEncoder, self).__init__()
        self.whole_cfg = cfg
        self.name = name
        self.example_key = example_key
        self.cfg = self.whole_cfg.model[name + '_encoder']
        self.encode_modules = nn.ModuleDict()

        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'frequency':
                self.encode_modules[k] = FrequencyEncoder(embedding_dim=item['embedding_dim'], )            
            elif item['arc'] == 'gaussian':
                self.encode_modules[k] = GaussianEncoder(embedding_dim=item['embedding_dim'], )                
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'unsqueeze':
                self.encode_modules[k] = UnsqueezeEncoder(norm_value=item['norm_value'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError

        self.embedding_dim = self.cfg.embedding_dim
        self.encoder_cfg = self.cfg.encoder
        self.encode_layers = MLP(in_channels=self.encoder_cfg.input_dim,
                                 hidden_channels=self.encoder_cfg.hidden_dim,
                                 out_channels=self.embedding_dim,
                                 layer_num=self.encoder_cfg.layer_num,
                                 layer_fn=fc_block,
                                 activation=self.encoder_cfg.activation,
                                 norm_type=self.encoder_cfg.norm_type,
                                 use_dropout=False)
        # self.activation_type = self.cfg.activation

        self.transformer_cfg = self.cfg.transformer
        self.transformer = Transformer(
            n_heads=self.transformer_cfg.head_num,
            embedding_size=self.embedding_dim,
            ffn_size=self.transformer_cfg.ffn_size,
            n_layers=self.transformer_cfg.layer_num,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation=self.transformer_cfg.activation,
            variant=self.transformer_cfg.variant,
        )
        self.output_cfg = self.cfg.output
        self.output_fc = fc_block(self.embedding_dim,
                                  self.output_cfg.output_dim,
                                  norm_type=self.output_cfg.norm_type,
                                  activation=self.output_cfg.activation)

    def forward(self, x):
        embeddings = []
        num = x[self.name + '_num']
        mask = sequence_mask(num, max_len=x[self.example_key].shape[1])

        for key, item in self.cfg.modules.items():
            assert key in x, f"transformer {self.name}: {key} not implemented"
            x_input = x[key]
            embeddings.append(self.encode_modules[key](x_input))

        x = torch.cat(embeddings, dim=-1)
        x = self.encode_layers(x)
        x = self.transformer(x, mask=mask)
        out_embedding = x
        info = self.output_fc(x.sum(dim=1) / (1e-8+num.unsqueeze(dim=-1)))
        return info, out_embedding, num


class SpatialEncoder(nn.Module):
    def __init__(self, cfg=None) -> None:
        super(SpatialEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.spatial_encoder
        self.activation_type = self.cfg.activation
        self.norm_type = self.cfg.norm_type

        self.project = conv2d_block(self.cfg.input_dim, self.cfg.project_dim, kernel_size=1,stride=1, padding=0, activation=self.activation_type,
                                     norm_type=self.norm_type)

        self.resnet_cfg = self.cfg.resnet
        self.get_resnet_blocks()
        self.spatial_size = reversed(self.whole_cfg.game.depthmap.resolution)  # reversed([64,32])

        self.output_cfg = self.cfg.output
        self.output_fc = fc_block(
            in_channels=self.get_outfc_input_size(),
            out_channels=self.output_cfg.output_dim,
            norm_type=self.output_cfg.norm_type,
            activation=self.output_cfg.activation)

    def get_resnet_blocks(self):
        layers = []
        dims = [self.cfg.project_dim] + self.resnet_cfg.down_channels
        for i in range(len(dims) - 1):
            layer = conv2d_block(in_channels=dims[i],
                                  out_channels=dims[i + 1],
                                  kernel_size=4,
                                  stride=2,
                                  padding=1,
                                  activation=self.activation_type,
                                  norm_type=self.norm_type,
                                  bias=False,
                                  )
            layers.append(layer)
            layers.append(ResBlock2(in_channels=dims[i + 1], ))
        self.resnet = torch.nn.Sequential(*layers)

    def get_outfc_input_size(self):
        with torch.no_grad():
            fake_data = torch.zeros(size=(1,1,*self.spatial_size))
            y = self.project(fake_data)
            y = self.resnet(y)
        return np.prod(torch.flatten(y).shape)

    def forward(self, x):
        out = self.project(x)
        out = self.resnet(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.output_fc(out)
        return out
    
class HeatMapEncoder(nn.Module):
    def __init__(self, cfg=None) -> None:
        super(HeatMapEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.heatmap_encoder
        self.activation_type = self.cfg.activation
        self.norm_type = self.cfg.norm_type
        
        self.kernel_size = 0
        for k,v in self.whole_cfg.env.heat_map.get('heat_map_deltas', {}).items():
            if k != "bottom2top_10":
                self.kernel_size += len(v)
        # self.kernel_size += 1
        
        self.resnet_cfg = self.cfg.resnet

        self.spatial_size = self.whole_cfg.env.heat_map.heat_map_size  # reversed([300,300])
        
        
        self.cnn1 = self.get_cnn()
        self.cnn2 = self.get_cnn()
        self.cnn3 = self.get_cnn()


        self.output_cfg = self.cfg.output
        self.output_fc1 = self.get_fc()
        
        self.output_fc2 = self.get_fc()
        
        self.output_fc3 = self.get_fc()
        
    def get_cnn(self):
        layers = [
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ]
        return torch.nn.Sequential(*layers)

        
    def get_project(self):
        return conv2d_block(1, self.cfg.project_dim, kernel_size=1,stride=1, padding=0, activation=self.activation_type,
                                     norm_type=self.norm_type)
    
    def get_fc(self):
        return fc_block(
            in_channels=self.get_outfc_input_size(),
            out_channels=self.output_cfg.output_dim,
            norm_type=self.output_cfg.norm_type,
            activation=self.output_cfg.activation)
    
    def get_resnet_blocks(self):
        layers = []
        dims = [self.cfg.project_dim] + self.resnet_cfg.down_channels
        for i in range(len(dims) - 1):
            layer = conv2d_block(in_channels=dims[i],
                                  out_channels=dims[i + 1],
                                  kernel_size=4,
                                  stride=2,
                                  padding=1,
                                  activation=self.activation_type,
                                  norm_type=self.norm_type,
                                  bias=False,
                                  )
            layers.append(layer)
            layers.append(ResBlock2(in_channels=dims[i + 1], ))
            layer2 = conv2d_block(in_channels=dims[i + 1],
                                  out_channels=dims[i + 1],
                                  kernel_size=4,
                                  stride=2,
                                  padding=1,
                                  activation=self.activation_type,
                                  norm_type=self.norm_type,
                                  bias=False,
                                  )
            layers.append(layer2)
        return torch.nn.Sequential(*layers)


    def get_outfc_input_size(self):
        with torch.no_grad():
            fake_data = torch.zeros(size=(1,1,*self.spatial_size))
            y = self.cnn1(fake_data)
           
        return np.prod(y.shape)

    def forward(self, x, need_weight = False):

        heatmap_out_info = {
            "heatmap_1":[0,0,0],
            "heatmap_2":[0,0,0],
            "heatmap_3":[0,0,0],
        }

        
        out1 = self.cnn1(x[:,0:1,...])
        with torch.no_grad():
            heatmap_out_info["heatmap_1"][0] = out1.abs().mean()
        out1 = self.output_fc1(out1)
        with torch.no_grad():
            heatmap_out_info["heatmap_1"][2] = out1.abs().mean()
        

        out2 = self.cnn2(x[:,1:2,...])
        with torch.no_grad():
            heatmap_out_info["heatmap_2"][0] = out2.abs().mean()
        out2 = self.output_fc2(out2)
        with torch.no_grad():
            heatmap_out_info["heatmap_2"][2] = out2.abs().mean()


        out3 = self.cnn3(x[:,2:3,...])
        with torch.no_grad():
            heatmap_out_info["heatmap_3"][0] = out3.abs().mean()

        out3 = self.output_fc3(out3)
        with torch.no_grad():
            heatmap_out_info["heatmap_3"][2] = out3.abs().mean()

        out = (out1 + out2 + out3)/3.

        with torch.no_grad():
            heatmap_out_info["heatmap_final"] = out.abs().mean()
        
        return out,heatmap_out_info
    
    
    
class MaxPool1DEncoder(nn.Module):
    def __init__(self, cfg,name:str):
        super(MaxPool1DEncoder, self).__init__()
        self.whole_cfg = cfg
        self.name = name
        self.cfg = self.whole_cfg.model[name + '_encoder']
        self.encode_modules = nn.ModuleDict()

        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'frequency':
                self.encode_modules[k] = FrequencyEncoder(embedding_dim=item['embedding_dim'], )
            elif item['arc'] == 'gaussian':
                self.encode_modules[k] = GaussianEncoder(embedding_dim=item['embedding_dim'], )                
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'unsqueeze':
                self.encode_modules[k] = UnsqueezeEncoder(norm_value=item['norm_value'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError
        self.layers = MLP(in_channels=self.cfg.input_dim, hidden_channels=self.cfg.hidden_dim,
                          out_channels=self.cfg.output_dim,
                          layer_num=self.cfg.layer_num,
                          layer_fn=fc_block,
                          activation=self.cfg.activation,
                          norm_type=self.cfg.norm_type,
                          use_dropout=False
                          )

        
    def forward(self, x: Dict[str, Tensor]):
        embeddings = []
        for key, item in self.cfg.modules.items():
            assert key in x, key
            embeddings.append(self.encode_modules[key](x[key]))

        out = torch.cat(embeddings, dim=-1)
        out = self.layers(out)
        out = torch.max(out,dim=-2)[0]
        
        return out

 
class MaxPool1DEncoderMultipleOutput(nn.Module):
    def __init__(self, cfg,name:str):
        super(MaxPool1DEncoderMultipleOutput, self).__init__()
        self.whole_cfg = cfg
        self.name = name
        self.cfg = self.whole_cfg.model[name + '_encoder']
        self.encode_modules = nn.ModuleDict()

        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'frequency':
                self.encode_modules[k] = FrequencyEncoder(embedding_dim=item['embedding_dim'], )
            elif item['arc'] == 'gaussian':
                self.encode_modules[k] = GaussianEncoder(embedding_dim=item['embedding_dim'], )                
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'unsqueeze':
                self.encode_modules[k] = UnsqueezeEncoder(norm_value=item['norm_value'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError
        self.layers = MLP(in_channels=self.cfg.input_dim, hidden_channels=self.cfg.hidden_dim,
                          out_channels=self.cfg.output_dim,
                          layer_num=self.cfg.layer_num,
                          layer_fn=fc_block,
                          activation=self.cfg.activation,
                          norm_type=self.cfg.norm_type,
                          use_dropout=False
                          )
        
    def forward(self, x: Dict[str, Tensor]):
        embeddings = []
        for key, item in self.cfg.modules.items():
            assert key in x, key
            embeddings.append(self.encode_modules[key](x[key]))

        out = torch.cat(embeddings, dim=-1)
        out = self.layers(out)
        out_maxpool = torch.max(out,dim=-2)[0]
        
        return out, out_maxpool 

class FlattenEncoder(nn.Module):
    def __init__(self,cfg,name: str):
        super(FlattenEncoder, self).__init__()
        self.whole_cfg = cfg
        self.name = name
        self.cfg = self.whole_cfg.model[name + '_encoder']
        self.encode_modules = nn.ModuleDict()
        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'frequency':
                self.encode_modules[k] = FrequencyEncoder(embedding_dim=item['embedding_dim'], )
            elif item['arc'] == 'gaussian':
                self.encode_modules[k] = GaussianEncoder(embedding_dim=item['embedding_dim'], )                
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'unsqueeze':
                self.encode_modules[k] = UnsqueezeEncoder(norm_value=item['norm_value'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError
        self.layers = MLP(in_channels=self.cfg.input_dim, hidden_channels=self.cfg.hidden_dim,
                          out_channels=self.cfg.output_dim,
                          layer_num=self.cfg.layer_num,
                          layer_fn=fc_block,
                          activation=self.cfg.activation,
                          norm_type=self.cfg.norm_type,
                          use_dropout=False
                          )
    def forward(self, x: Dict[str, Tensor]):
        embeddings = []
        for key, item in self.cfg.modules.items():
            assert key in x, key
            embeddings.append(self.encode_modules[key](x[key]))

        out = torch.cat(embeddings, dim=-1)
        out = torch.flatten(out,start_dim=-2)
        out = self.layers(out)
        return out

    


class Encoder(nn.Module):
    def __init__(self, cfg,):
        super(Encoder, self).__init__()
        self.whole_cfg = cfg

        
        
        self.scalar_encoder = ScalarEncoder(cfg,name='scalar')
        self.teammate_encoder = FlattenEncoder(cfg,name='teammate')
        self.backpack_encoder = MaxPool1DEncoder(cfg,name="backpack_item")
        self.player_weapon_encoder = FlattenEncoder(cfg,name="player_weapon")
        self.door_encoder = MaxPool1DEncoder(cfg,name="door")
        self.only_v_encoder = MaxPool1DEncoder(cfg,name="only_v")
        self.supply_encoder = MaxPool1DEncoder(cfg,name="supply")
        self.enemy_encoder = MaxPool1DEncoder(cfg,name="enemy")
        self.enemy_visible_encoder = MaxPool1DEncoderMultipleOutput(cfg,name="enemy_visible")
        # self.spatial_encoder = SpatialEncoder(cfg)
        self.heatmap_encoder = HeatMapEncoder(cfg)
        self.history_position_encoder = HistoryPositionEncoder(cfg)
        self.monster_encoder = MaxPool1DEncoder(cfg,name="monster")
        self.event_encoder = MaxPool1DEncoder(cfg,name="event")
        self.rotation_encoder = MaxPool1DEncoder(cfg,name="rotation")

        
        
    def forward(self, x):
        # heat_map = self.process_heat_map(x["heatmap_info"])
        # xclone= copy.deepcopy(x)
        # x=x['model_input']
        heat_map_info,heatmap_out_info = self.heatmap_encoder(x["heatmap_info"])
        backpack_info = self.backpack_encoder(x["backpack_item_info"])
        weapon_info = self.player_weapon_encoder(x["player_weapon_info"])
        door_info = self.door_encoder(x["door_info"])
        monster_info = self.monster_encoder(x["monster_info"])
        # spatial_info = self.spatial_encoder(x['spatial_info'])
        only_v_embedding = self.only_v_encoder(x['only_v_info'])
        scalar_info = self.scalar_encoder(x["scalar_info"])
        teammate_info = self.teammate_encoder(x["teammate_info"])
        enemy_info = self.enemy_encoder(x["enemy_item_info"])
        enemy_visible_no_maxpool_info,enemy_visible_maxpool_info = self.enemy_visible_encoder(x["visible_enemy_item_info"])
        supply_info = self.supply_encoder(x["supply_item_info"])
        event_info = self.event_encoder(x["event_info"])
        rotation_info = self.rotation_encoder(x["rotation_info"])
        
        history_positions_info = self.history_position_encoder(x["history_positions_info"])
        
        embedding = torch.cat([scalar_info,backpack_info,weapon_info,teammate_info, enemy_info,supply_info, door_info,heat_map_info,history_positions_info,monster_info,event_info,rotation_info,enemy_visible_maxpool_info], dim=1)
        visible_enemy_num = x["visible_enemy_item_info"]["enemy_item_num"]
        visible_enemy_distance = x["visible_enemy_item_info"]["distance"]
        return embedding,only_v_embedding,heatmap_out_info, enemy_visible_no_maxpool_info, visible_enemy_num, visible_enemy_distance
