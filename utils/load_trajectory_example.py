import os
import sys
import math
current_file_path = os.path.abspath(__file__)

from click import progressbar
from cv2 import sort
from utils.config_helper import read_config
from utils.model.model import Model,Model_expand
from utils.features import Features
from utils.features_nonorm import Features_NoNorm
from utils.collate_fn import default_collate_with_dim
from utils.heatmap_utils.heatmap_helper import MoreMap, load_heatmap,HeatMap,load_more_map
from utils.env_info.supply_item_info import all_supply_items
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import os
import time
import random
import json
import torch.nn as nn
from collate_fn import default_collate_with_dim

class Trajectory(nn.Module):
    def __init__(self,root_path,config_path, gpu_id=None,for_llm=False) -> None:   
        super().__init__()
        self.for_llm=for_llm
        if for_llm:
            self.ModelClass = Model_expand
            # self.ModelClass = Model_Plus
        else:
            self.ModelClass = Model         
        self.supply_item_info = all_supply_items
        self.mapsize = [403300,403300,33000]
        self.position_norm = 100
        self.whole_cfg = read_config(config_path)
        self.heatmap_data = load_heatmap(os.path.join('./', self.whole_cfg.env.heat_map.heat_map_path))
        more_map_param = load_more_map()
        self.more_map_class = MoreMap(more_map_param,self.whole_cfg)
        self.features = Features(self.whole_cfg, heat_map_data = self.heatmap_data )
        self.features_nonorm = Features_NoNorm(self.whole_cfg, heat_map_data = self.heatmap_data )
        self.doors_pos_matrix = None
        self.doors_visble_distance =   3000 
        self.model = self.ModelClass(self.whole_cfg)
        
        output2action_dict_old = torch.load('/mnt/nfs2/aaa/Process_Replay/output2action.pt')
        self.output2action_dict_old = output2action_dict_old['single_head']
 
    def load_model(self,root_path='/home/ccc1/nfs/ccc/ccc_nfs/MMD-project/llava/model/rlmodal_encoder/load_state'):
        ckpts = torch.load(os.path.join(root_path,self.whole_cfg.var2), map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpts['model'])
        
 
    def preprocess_obs(self,state,objs,depth_map,last_state,last_action,info,search_target_player_id):        
        obs = self.features.transform_obs(state=state, objs=objs,last_state=last_state, last_action=last_action, info=info, depth_map=depth_map, search_target_player_id = search_target_player_id)
        obs = default_collate_with_dim([obs], ) 
        return obs
    
    def preprocess_obs_nonorm(self,state,objs,depth_map,last_state,last_action,info,search_target_player_id):        
        obs = self.features_nonorm.transform_obs(state=state, objs=objs,last_state=last_state, last_action=last_action, info=info, depth_map=depth_map, search_target_player_id = search_target_player_id)
        # obs1 = default_collate_with_dim([obs], )
 
        return obs    
    
    def distance_via_id(self, id_1, id_2,info=None):
        if(info is None):
            info=self.info
        x1 = info['player_state'][id_1].state.position.x
        y1 = info['player_state'][id_1].state.position.y
        z1 = info['player_state'][id_1].state.position.z
        x2 = info['player_state'][id_2].state.position.x
        y2 = info['player_state'][id_2].state.position.y
        z2 = info['player_state'][id_2].state.position.z
        return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    
    def find_search_target(self, info, self_id ):
        info_state = info['player_state']
        distance_list = []
        id_list = []
        for id,player_state in info_state.items():
            if id!=self_id and  player_state.state.team_id != info_state[int(self_id)].state.team_id and player_state.state.alive_state!=2:   ### find nearest enemy
                distance2enemy = self.distance_via_id(self_id, id,info)
                id_list.append(id)
                distance_list.append(distance2enemy)
        if  len(distance_list)==0:
            return None
        # if min(distance_list) <= 10000: # 近距离也暴露
        #     return None
        nearest_index = distance_list.index(min(distance_list))
        nearest_id = id_list[nearest_index]
        return nearest_id


    def load_state_onestep(self, trajectory_path=None, json_data=None,root_json_data=None): 
        # extends=json_data["Actions Extend Nums"]   
        trajectory = torch.load(trajectory_path)
        # if len(extends)==2:
        #     start_idx = max(extends[0],0)
        #     end_idx = min(len(trajectory)-extends[-1], len(trajectory))
        #     trajectory=trajectory[start_idx:end_idx]
 
        # random_idxs = list(range(len(trajectory)))
        # random.shuffle(random_idxs)
        for traj in trajectory:
            # traj = trajectory[i]
            self.src_action = traj[1]
            traj_input = traj[0]
            self.state = traj_input['state']
            self.objs = traj_input['objs']
            depth_map = traj_input['depth_map']
            last_state = traj_input['last_state']
            self.last_action = traj_input['last_action']
            self.info = traj_input['info']
            if 'enemy_id_queue' in traj_input.keys():
                self.features.enemy_id_queue = traj_input['enemy_id_queue']
            if 'enemy_info_queue' in traj_input.keys():
                self.features.enemy_info_queue = traj_input['enemy_info_queue']
            if 'ENEMY_SEE_INFO' in traj_input.keys():
                self.features.ENEMY_SEE_INFO = traj_input['ENEMY_SEE_INFO']
            if 'self_history_positions' in traj_input.keys():
                self.features.self_history_positions = traj_input['self_history_positions']
            if 'history_event_info' in traj_input.keys():
                self.features.history_event_info = traj_input['history_event_info']

            # self.search_target_player_id = self.find_search_target(self.info,player_id)
            if 'search_target_player_id' in traj_input.keys():
                self.search_target_player_id = traj_input['search_target_player_id']
            else:
                self.search_target_player_id = self.find_search_target(self.info,self.features.id)
            model_input = self.preprocess_obs(state= self.state,
                                            objs= self.objs,
                                            depth_map= depth_map,
                                            last_state= last_state,
                                            last_action= self.last_action,
                                            info=self.info,
                                            search_target_player_id=self.search_target_player_id)
            # self.get_action(model_input)
 
        return model_input

    def transform_door_info(self,doors):

        
        # doors = info['doors']
 
        
        # 预先将上千道门的位置信息进行储存，后续使用矩阵运算筛选出可见距离内的门！
        if self.doors_pos_matrix is None:
            self.doors_pos_matrix = []
            self.doors_categorys = []
            for door_id, door in doors.items():
                self.doors_categorys.append(door_id)
                self.doors_pos_matrix.append([door.position.x,door.position.y,door.position.z])
            self.doors_pos_matrix = torch.as_tensor(self.doors_pos_matrix)
            self.doors_categorys = torch.as_tensor(self.doors_categorys)

        return self.doors_pos_matrix


    def load_trajectory_onestep(self, trajectory_path=None, json_data=None,root_json_data=None,load_pt=False, pt_content=None): 
        # extends=json_data["Actions Extend Nums"]   
        if not load_pt:
            trajectory = torch.load(trajectory_path)
        else:
            trajectory = pt_content
        # if len(extends)==2:
        #     start_idx = max(extends[0],0)
        #     end_idx = min(len(trajectory)-extends[-1], len(trajectory))
        #     trajectory=trajectory[start_idx:end_idx]
 
        # random_idxs = list(range(len(trajectory)))
        # random.shuffle(random_idxs)
        if isinstance(trajectory,list):
            for traj in trajectory:
                # traj = trajectory[i]
                self.src_action = traj[1]
                traj_input = traj[0]
                self.state = traj_input['state']
                self.objs = traj_input['objs']
                depth_map = traj_input['depth_map']
                last_state = traj_input['last_state']
                self.last_action = traj_input['last_action']
                self.info = traj_input['info']
                if 'enemy_id_queue' in traj_input.keys():
                    self.features.enemy_id_queue = traj_input['enemy_id_queue']
                if 'enemy_info_queue' in traj_input.keys():
                    self.features.enemy_info_queue = traj_input['enemy_info_queue']
                if 'ENEMY_SEE_INFO' in traj_input.keys():
                    self.features.ENEMY_SEE_INFO = traj_input['ENEMY_SEE_INFO']
                if 'self_history_positions' in traj_input.keys():
                    self.features.self_history_positions = traj_input['self_history_positions']
                if 'history_event_info' in traj_input.keys():
                    self.features.history_event_info = traj_input['history_event_info']

                # self.search_target_player_id = self.find_search_target(self.info,player_id)
                if 'search_target_player_id' in traj_input.keys():
                    self.search_target_player_id = traj_input['search_target_player_id']
                else:
                    self.search_target_player_id = self.find_search_target(self.info,self.features.id)
                model_input = self.preprocess_obs(state= self.state,
                                                objs= self.objs,
                                                depth_map= depth_map,
                                                last_state= last_state,
                                                last_action= self.last_action,
                                                info=self.info,
                                                search_target_player_id=self.search_target_player_id)
        elif isinstance(trajectory,dict):
            traj_input = trajectory
            self.state = traj_input['state']
            self.objs = traj_input['objs']
            depth_map = traj_input['depth_map']
            last_state = traj_input['last_state']
            self.last_action = traj_input['last_action']
            self.info = traj_input['info']
            if 'enemy_id_queue' in traj_input.keys():
                self.features.enemy_id_queue = traj_input['enemy_id_queue']
            if 'enemy_info_queue' in traj_input.keys():
                self.features.enemy_info_queue = traj_input['enemy_info_queue']
            if 'ENEMY_SEE_INFO' in traj_input.keys():
                self.features.ENEMY_SEE_INFO = traj_input['ENEMY_SEE_INFO']
            if 'self_history_positions' in traj_input.keys():
                self.features.self_history_positions = traj_input['self_history_positions']
            if 'history_event_info' in traj_input.keys():
                self.features.history_event_info = traj_input['history_event_info']

            # self.search_target_player_id = self.find_search_target(self.info,player_id)
            if 'search_target_player_id' in traj_input.keys():
                self.search_target_player_id = traj_input['search_target_player_id']
            else:
                self.search_target_player_id = self.find_search_target(self.info,self.features.id)
            model_input = self.preprocess_obs(state= self.state,
                                            objs= self.objs,
                                            depth_map= depth_map,
                                            last_state= last_state,
                                            last_action= self.last_action,
                                            info=self.info,
                                            search_target_player_id=self.search_target_player_id)

            # self.get_action(model_input)
 
        return model_input


    def describe_circle_state(self,features_info):  
        own_id = self.features.id      
        own_player_state = self.info['player_state'][own_id]
 
        # pos_x, pos_y, pos_z = own_player_state.state.position.x / self.position_norm
 
        alive_state_dict={0:'normal status',1:'be knocked down, wait for help',2:'dead'}
        alive_state = alive_state_dict[own_player_state.state.alive_state]

        safety_area = self.state.safety_area 
        pos_x = safety_area.center.x/self.position_norm 
        pos_y = safety_area.center.y/self.position_norm 
        radius = safety_area.radius/self.position_norm 
        safezone_pain = safety_area.pain
        safezone_appear_time = safety_area.safezone_appear_time
        safezone_delay_time = safety_area.delay_time
        own_player_in_blue_safetyarea = self.player_in_safetyarea(own_player_state.state.position.x,
                                                        own_player_state.state.position.y,
                                                        safety_area.center.x,
                                                        safety_area.center.y,
                                                        safety_area.radius)
        state_dict ={
            'safety_area_position':(pos_x, pos_y),
            'safety_area_radius':radius,
            'safety_area_pain':safezone_pain,
            'is_in_safety_area':own_player_in_blue_safetyarea,
            'safety_area_appear_time':safezone_appear_time,            
            'safety_area_delay_time':safezone_delay_time,
        }

        return state_dict

    def computer_distance(self, x, y, z, target_x, target_y, target_z):
        return math.sqrt((target_x - x)**2 + (target_y - y)**2 + (target_z-z)**2)
 

    def player_in_safetyarea(self, player_x, player_y, circle_x, circle_y, radius):
        return self.computer_distance(player_x, player_y,0, circle_x, circle_y,0) < radius
 

    def calculate_move_angle(self, x, y):
        angle = math.degrees(math.atan2(y, x))
        if angle < 0:
            angle += 360
        return angle
    
    def describe_myself_state(self,features_info):  
        own_id = self.features.id      
        own_player_state = self.info['player_state'][own_id]
        team_id = own_player_state.state.team_id
        pos_x, pos_y, pos_z = own_player_state.state.position.x,own_player_state.state.position.y,own_player_state.state.position.z
        rot_x, rot_y, rot_z = own_player_state.state.rotation.x,own_player_state.state.rotation.y,own_player_state.state.rotation.z
        speed_x, speed_y, speed_z = own_player_state.state.speed.x,own_player_state.state.speed.y,own_player_state.state.speed.z
        speed_x = round(speed_x/self.position_norm,)
        speed_y = round(speed_y/self.position_norm,)
        speed_z = round(speed_z/self.position_norm,)
        hp=own_player_state.state.hp
        if hp<30:
            hp_str = f"{int(hp)} (low)"
        elif hp<90:
            hp_str = f"{int(hp)} (middle)"
        else:
            hp_str = f"{int(hp)} (high)"
        body_state=own_player_state.state.body_state
        my_body_state = None
        if body_state==1:
            my_body_state= 'jumping'
        elif body_state==2:
            my_body_state= 'crouched'
        elif body_state==6:
            my_body_state= 'swimming'
        elif body_state==8:
            my_body_state= 'prone'

        pos={}
        pos["position_x"]=int(pos_x/10)
        pos["position_y"]=int(pos_y/10)
        water_if, house_if = self.more_map_class.check_pos(pos)
        in_water,in_house = None, None
        if water_if:
            in_water = 'true'
        if house_if:
            location_place = 'indoor'
        else:
            location_place='outdoor'
        if np.abs(speed_x)<0.001 and np.abs(speed_y)<0.001:
            move_direction = 'none'
        else:
            move_direction = f"yaw_{int(self.calculate_move_angle(speed_x, speed_y))}"
        oxygen=own_player_state.state.oxygen
        alive_state_dict={0:'normal status',1:'be knocked down, wait for help, unable to fight',2:'dead'}
        alive_state = alive_state_dict[own_player_state.state.alive_state]
        progress_bar = own_player_state.progress_bar
        progress_bar_key = [
                            'heal myself', # 是否在打药包
                            'remaining time to heal myself', # 剩余多久打完药包
                            'help up teammate', # 是否在救队友
                            'remaining time to help up teammate', # 剩余多久救完队友
                            'be helped by teammate', # 是否在被队友扶
                            'remaining time to be helped up', # 剩余多久被扶起来
                            'reload bullet', # 是否在换弹
                            'remaining time to reload bullet', # 剩余多久换好子弹
                            ]
 
        if progress_bar.type != 0:
            progress_item_value = progress_bar_key[2*(progress_bar.type-1)] 
            progress_item_remain_key =  progress_bar_key[2*(progress_bar.type-1)+1] 
            progress_remain_value_time = round(progress_bar.remain_time,1)
        
        can_see_teammate,can_see_enemy = 'false','false'
        if len(own_player_state.visble_player_ids)>0:
            for p_id in  own_player_state.visble_player_ids:
                if self.info['player_state'][p_id].state.alive_state==2:
                    continue
                if self.info['player_state'][p_id].state.team_id == team_id:
                    can_see_teammate='true'
                else:
                    can_see_enemy='true'

        scarlar_info = features_info['scalar_info']
        
        scalar_dict ={
            # 'player_id':own_id,
            # 'team_id':team_id,
            'health_point':hp_str,
            'alive_state':alive_state, # 正常活动；被击倒，待救援；死亡 daijiuyuan, dead
            # 'oxygen_in_water':own_player_state.state.oxygen, #潜水时的氧气剩余值             
            'position':(round(pos_x/self.position_norm,),round(pos_y/self.position_norm, ),round(pos_z/self.position_norm, )),
            'speed':(speed_x, speed_y, speed_z),
            'move_direction':f"{move_direction}",
            'view_direction':f"yaw_{int(rot_z)}",  
            'position_type':f"{location_place}",
            # 'rotation':{'roll':int(rot_x),'pitch':int(rot_y),'yaw':int(rot_z)},     
            # 'can_see_enemy':can_see_enemy,
            # 'can_see_teammate':can_see_teammate,        
            "how_far_obstacles_are_towards_4_yaw_directions (format is  yaw_degree:distance)":{
                                                    'yaw_0':round(scarlar_info['dir_distance_1'].item(),1),
                                                    'yaw_90':round(scarlar_info['dir_distance_5'].item(),1),                                                    
                                                    'yaw_180':round(scarlar_info['dir_distance_9'].item(),1),                                                
                                                    'yaw_270':round(scarlar_info['dir_distance_13'].item(),1)
                                                    },  
            # "distance_to_obstacles_towards_16_yaw_directions (format is  yaw_degree:distance)":{'yaw_0':round(scarlar_info['dir_distance_1'].item(),1),'yaw_22':round(scarlar_info['dir_distance_2'].item(),1),'yaw_45':round(scarlar_info['dir_distance_3'].item(),1),
            #                                         'yaw_67':round(scarlar_info['dir_distance_4'].item(),1),'yaw_90':round(scarlar_info['dir_distance_5'].item(),1),'yaw_112':round(scarlar_info['dir_distance_6'].item(),1),
            #                                         'yaw_135':round(scarlar_info['dir_distance_7'].item(),1),'yaw_157':round(scarlar_info['dir_distance_8'].item(),1),'yaw_180':round(scarlar_info['dir_distance_9'].item(),1),
            #                                         'yaw_202':round(scarlar_info['dir_distance_10'].item(),1),'yaw_225':round(scarlar_info['dir_distance_11'].item(),1),'yaw_247':round(scarlar_info['dir_distance_12'].item(),1),
            #                                         'yaw_270':round(scarlar_info['dir_distance_13'].item(),1),'yaw_292':round(scarlar_info['dir_distance_14'].item(),1),'yaw_315':round(scarlar_info['dir_distance_15'].item(),1),
            #                                         'yaw_337':round(scarlar_info['dir_distance_16'].item(),1)},      
        }
        if own_player_state.state.is_running:
            scalar_dict['is_running']='true' ###可以用来过滤mov_dir_xx这类动作
        if my_body_state is not None:
            scalar_dict['body_state']=my_body_state
        if in_water == 'true':
            scalar_dict['in_the_water']='true'
            scalar_dict['oxygen']= int(oxygen)           
        if own_player_state.state.is_aiming:
            scalar_dict['is_aiming_enemy']='true'
        if own_player_state.state.is_firing:
            scalar_dict['is_firing']='true'
        if can_see_enemy=='true':
            scalar_dict['can_see_enemy']=can_see_enemy
        if can_see_teammate=='ture':
            scalar_dict['can_see_teammate']=can_see_teammate
        # if own_player_state.state.is_pose_changing:
        #     scalar_dict['is_pose_changing']=1
        # if own_player_state.state.is_pose_changing:
        #     scalar_dict['is_pose_changing']=1
        # if own_player_state.state.is_pose_changing:
        #     scalar_dict['is_pose_changing']=1
        # 'is_running':own_player_state.state.is_running,
        # 'is_aiming_enemy':own_player_state.state.is_aiming,
        # 'is_firing':own_player_state.state.is_firing,
        # 'is_holding_weapon':own_player_state.state.is_holding,
        # 'is_falling_off':own_player_state.state.is_falling,
        # 'is_picking_supply':own_player_state.state.is_picking, 
        if progress_bar.type != 0:
            scalar_dict['progress status']=progress_item_value
            scalar_dict[progress_item_remain_key]=progress_remain_value_time

        # state_str_list = []
        # for k,v in scalar_dict.items():
        #     state_str_list.append(f"{k}:{v}")
        
        # state_str = '; '.join(state_str_list)
        # state_str = 'my state are:'+state_str
        return scalar_dict


    def describe_enemy_state(self, ):  
        own_id = self.features.id      
        own_player_state = self.info['player_state'][own_id]
        team_id = own_player_state.state.team_id
        own_position= own_player_state.state.position
        alive_state_dict={0:'normal status',1:'be knocked down, unable to fight back',2:'dead'}
        state_info = self.info['player_state']   
        nearest_enemy_id = None
        nearest_enemy_dist = np.inf
        alive_enemy_num = 0
        for player_id, player_item in state_info.items():
            if not self.info['alive_players'][player_id]:
                continue
            if (player_id == own_id or state_info[player_id].state.team_id == team_id):
                continue
            alive_enemy_num += 1

            enemy_pos = player_item.state.position
            enemy_distance = math.sqrt((own_position.x-enemy_pos.x)**2 + (own_position.y-enemy_pos.y)** 2 + (own_position.z-enemy_pos.z)** 2)
            if nearest_enemy_dist>enemy_distance:
                nearest_enemy_dist=enemy_distance
                nearest_enemy_id=player_id
        if nearest_enemy_id is None:
            state_dict ={
                'alive_enemy_num':alive_enemy_num
            }
            # state_str_list = []
            # for k,v in state_dict.items():
            #     state_str_list.append(f"{k}:{v}")
            # state_str = '; '.join(state_str_list)
            # state_str = 'Nearest enemy state are:'+state_str
            return state_dict
        nearest_enemy_state = state_info[nearest_enemy_id]
        enemy_team_id = nearest_enemy_state.state.team_id
        enemy_pos = nearest_enemy_state.state.position
        pos_x, pos_y, pos_z = enemy_pos.x/self.position_norm, enemy_pos.y/self.position_norm, enemy_pos.z/self.position_norm,
        pos={}
        pos["position_x"]=int(pos_x*10)
        pos["position_y"]=int(pos_y*10)
        water_if, house_if = self.more_map_class.check_pos(pos)
        in_water,in_house = None, None
        if water_if:
            in_water = 'true'
        if house_if:
            location_place = 'indoor'
        else:
            location_place='outdoor'        
        rel_pos_x,rel_pos_y,rel_pos_z = pos_x - own_position.x/self.position_norm, pos_y-own_position.y/self.position_norm, pos_z-own_position.z/self.position_norm        
        relative_direction = f"yaw_{int(self.calculate_move_angle(rel_pos_x, rel_pos_y))}"        
        distance = int(nearest_enemy_dist/self.position_norm)
        if distance<50:
            distance_str = f"{distance} (near)"
        elif distance<150:
            distance_str = f"{distance} (middle)"
        else:
            distance_str = f"{distance} (far)"        
        alive_state = alive_state_dict[nearest_enemy_state.state.alive_state]
        hp=nearest_enemy_state.state.hp
        if hp<30:
            hp_str = f"{int(hp)} (low)"
        elif hp<90:
            hp_str = f"{int(hp)} (middle)"
        else:
            hp_str = f"{int(hp)} (high)"
        if nearest_enemy_id  in own_player_state.visble_player_ids:
            i_can_see_enemy = 'true'
        else:
            i_can_see_enemy = 'false'
        if own_id in state_info[nearest_enemy_id].visble_player_ids:
            enemy_see_me = 'true'
        else:
            enemy_see_me = 'false' 
        if len(nearest_enemy_state.weapon.player_weapon)>0:
            hold_gun = 'true'
        else:
            hold_gun = 'false'
        state_dict ={         
            # 'player_id':nearest_enemy_id,
            # 'team_id':enemy_team_id,
            'health_point':hp_str,
            'hold_gun':hold_gun,
            'enemy_can_see_me':enemy_see_me,
            'i_can_see_enemy':i_can_see_enemy,         
            'alive_state':alive_state,    
            'distance_to_me':distance_str,         
            'position':(round(pos_x,),round(pos_y,),round(pos_z,)),
            'position_type':location_place,
            'relative_position_to_me':(round(rel_pos_x,),round(rel_pos_y,),round(rel_pos_z,)), 
            # 'direction_of_relative_position_to_me':relative_direction,
            'which_direction_enemy_is_located_to_me':relative_direction,
        }
        if in_water=='true':
            state_dict['in_water']='true'
        self.nearest_enemy_id = nearest_enemy_id
        # state_str_list = []
        # for k,v in state_dict.items():
        #     state_str_list.append(f"{k}:{v}")
        
        # state_str = '; '.join(state_str_list)
        # state_str = 'Nearest enemy states:'+state_str
        return state_dict

 

    def describe_teammate_state(self, ):        
        own_id = self.features.id      
        own_player_state = self.info['player_state'][own_id]
        team_id = own_player_state.state.team_id
        own_position= own_player_state.state.position
        alive_state_dict={0:'normal status',1:'be knocked down, wait for help, unable to fight',2:'dead'}

        state_info = self.info['player_state']   
        nearest_teammate_id = None
        nearest_teammate_dist = np.inf
        alive_teammate_num = 0
        for player_id, player_item in state_info.items():
            if not self.info['alive_players'][player_id]:
                continue
            if (player_id ==  own_id or state_info[player_id].state.team_id != team_id):
                continue
            alive_teammate_num += 1

            teammate_pos = player_item.state.position
            teammate_distance = math.sqrt((own_position.x-teammate_pos.x)**2 + (own_position.y-teammate_pos.y)** 2 + (own_position.z-teammate_pos.z)** 2)
            if nearest_teammate_dist>teammate_distance:
                nearest_teammate_dist=teammate_distance
                nearest_teammate_id=player_id
        if nearest_teammate_id is None:
            state_dict ={
                'alive_teammate_num':alive_teammate_num
            }
            # state_str_list = []
            # for k,v in state_dict.items():
            #     state_str_list.append(f"{k}:{v}")
            # state_str = '; '.join(state_str_list)
            # state_str = 'Nearest teammate states:'+state_str
            return state_dict
        nearest_teammate_state = state_info[nearest_teammate_id]
        teammate_team_id = nearest_teammate_state.state.team_id
        teammate_pos = nearest_teammate_state.state.position
        pos_x, pos_y, pos_z = teammate_pos.x/self.position_norm, teammate_pos.y/self.position_norm, teammate_pos.z/self.position_norm,
        pos={}
        pos["position_x"]=int(pos_x*10)
        pos["position_y"]=int(pos_y*10)
        water_if, house_if = self.more_map_class.check_pos(pos)
        in_water,in_house = None, None
        if water_if:
            in_water = 'true'
        if house_if:
            location_place = 'indoor'
        else:
            location_place='outdoor'           
        rel_pos_x,rel_pos_y,rel_pos_z = pos_x - own_position.x/self.position_norm, pos_y-own_position.y/self.position_norm, pos_z-own_position.z/self.position_norm
        relative_direction = f"yaw_{int(self.calculate_move_angle(rel_pos_x, rel_pos_y))}"   
        distance = int(nearest_teammate_dist/self.position_norm)
        if distance<50:
            distance_str = f"{distance} (near)"
        elif distance<150:
            distance_str = f"{distance} (middle)"
        else:
            distance_str = f"{distance} (far)"
        alive_state = alive_state_dict[nearest_teammate_state.state.alive_state]
        hp=nearest_teammate_state.state.hp
        if hp<30:
            hp_str = f"{int(hp)} (low)"
        elif hp<90:
            hp_str = f"{int(hp)} (middle)"
        else:
            hp_str = f"{int(hp)} (high)"
        if nearest_teammate_id  in own_player_state.visble_player_ids:
            i_can_see_teammate = 'true'
        else:
            i_can_see_teammate = 'false'
        if own_id in state_info[nearest_teammate_id].visble_player_ids:
            teammate_see_me = 'true'
        else:
            teammate_see_me = 'false'  
        if len(nearest_teammate_state.weapon.player_weapon)>0:
            hold_gun = 'true'
        else:
            hold_gun = 'false'
        state_dict ={         
            # 'player_id':nearest_teammate_id,
            # 'team_id':teammate_team_id,
            'health_point':hp_str,
            'hold_gun':hold_gun,
            'teammate_can_see_me':teammate_see_me,
            'i_can_see_teammate':i_can_see_teammate,         
            'alive_state':alive_state,    
            'distance_to_me':distance_str,         
            'position':(round(pos_x,),round(pos_y,),round(pos_z,)),
            'position_type':location_place,
            'relative_position_to_me':(round(rel_pos_x,),round(rel_pos_y,),round(rel_pos_z,)), 
            # 'direction_of_relative_position_to_me':relative_direction,
            'which_direction_teammate_is_located_to_me':relative_direction,
        }
        if in_water=='true':
            state_dict['in_water']='true'
        # state_str_list = []
        # for k,v in state_dict.items():
        #     state_str_list.append(f"{k}:{v}")
        
        # state_str = '; '.join(state_str_list)
        # state_str = 'Nearest teammate states:'+state_str
        return state_dict    


    def describe_weapon_backpack_state(self, ):       
        own_id = self.features.id      
        own_player_state = self.info['player_state'][own_id]
        
        ## weapons ##
        player_weapons = own_player_state.weapon.player_weapon
        weapons_number = len(player_weapons)
        weapon_dict = {0:'use no weapon',1:'right weapon', 2:'left weapon'}
        use_which_weapon = weapon_dict[own_player_state.state.active_weapon_slot]
        weapon_used = player_weapons[own_player_state.state.active_weapon_slot-1]
        current_bullet = weapon_used.bullet
        # weapon_remain_reloading = weapon_used.remain_reloading


        ## backpack ##
        RecoveryHP = [3301,3301]
        self.features.bullet2gun
        self.features.gun2bullet
        backpack_items = own_player_state.backpack.backpack_item
        have_medicine = 'false'
        have_bullets_in_bag = 'false'
        for backpack_item in backpack_items:
          
            if backpack_item.category in RecoveryHP:
                have_medicine = 'true'


            if backpack_item.category in self.features.bullet2gun.keys():                 
                if weapon_used.category in self.features.bullet2gun[backpack_item.category]:
                        have_bullets_in_bag = 'true'


    
        if own_player_state.state.active_weapon_slot==0:
            state_dict ={
                'use_which_weapon':use_which_weapon,            
                # 'numbers_of_available_weapons' :weapons_number ,
                'have_medicine_in_bag':have_medicine
            }            
        else:
            state_dict ={
                'use_which_weapon':use_which_weapon,            
                # 'numbers_of_available_weapons' :weapons_number,
                'current_bullet':current_bullet,
                # 'remain_reloading_time':round(weapon_remain_reloading/1000,1),
                'have_bullets_in_bag':have_bullets_in_bag,
                'have_medicine_in_bag':have_medicine

        }
        # state_str_list = []
        # for k,v in state_dict.items():
        #     state_str_list.append(f"{k}:{v}")
        
        # state_str = '; '.join(state_str_list)
        # state_str = 'Equipment states:'+state_str
        return state_dict      


    def describe_monster_state(self, ):      
        
        own_id = self.features.id      
        own_player_state = self.info['player_state'][own_id]

        own_position= own_player_state.state.position

        monsters = self.state.monsters
   
        is_monster_attacking_me = 0
        monsters_list=[]
        for monster in monsters:
            mon_position = monster.position
            mon_pos_x = mon_position.x
            mon_pos_y = mon_position.y
            mon_pos_z = mon_position.z
            mon2own_distance = self.computer_distance(mon_pos_x,mon_pos_y,mon_pos_z,own_position.x,own_position.y,own_position.z)
            if  mon2own_distance > self.monsters_visble_distance:
                continue
 
            
            mon_relative_me_pos_x = mon_position.x - own_position.x
            mon_relative_me_pos_y = mon_position.y - own_position.y
            mon_relative_me_pos_z = mon_position.z - own_position.z
 
            
   
            mon_target_id = monster.target_id
            ## TO DO QUDIAO
            if mon_target_id == own_id:
                is_monster_attacking_me = 1 # 自己
 
            
            monsters_list.append(
                [
                    mon2own_distance/self.position_norm,     
                    mon_pos_x/self.position_norm,
                    mon_pos_y/self.position_norm,
                    mon_pos_z/self.position_norm,
                    mon_relative_me_pos_x/self.position_norm,
                    mon_relative_me_pos_y/self.position_norm,
                    mon_relative_me_pos_z/self.position_norm,
                    is_monster_attacking_me
                ]
            )
        number_of_monsters = len(monsters_list)
        sorted_monsters_list = sorted(monsters_list)
        if len(monsters_list)>0:
            nearest_monster = sorted_monsters_list[0]
        if number_of_monsters==0:
            state_dict = {'number_of_monsters':0}
        else:
            distance = int(nearest_monster[0])
            if distance<50:
                distance_str = f"{distance} (near)"
            elif distance<150:
                distance_str = f"{distance} (middle)"
            else:
                distance_str = f"{distance} (far)"            
            state_dict ={         
                'number_of_monsters':number_of_monsters,   
                'distance_to_me':distance_str,           
                'position':(round(nearest_monster[1],),round(nearest_monster[2],),round(nearest_monster[3],)),
                'relative_position':(round(nearest_monster[4],),round(nearest_monster[5],),round(nearest_monster[6],)),
                'monster_attack_target':nearest_monster[-1]
            }
        # state_str_list = []
        # for k,v in state_dict.items():
        #     state_str_list.append(f"{k}:{v}")
        
        # state_str = '; '.join(state_str_list)
        # state_str = 'Monster states are:'+state_str
        return state_dict      

        
    def find_nearest_doors(self,pos):
        dis_matrix = (torch.as_tensor(pos) - self.doors_pos_matrix).norm(p=2,dim=-1)
        index = dis_matrix.le(self.doors_visble_distance)
        return self.features.doors_categorys[index]
    
    def describe_door_state(self, ):        
        own_id = self.features.id      
        own_player_state = self.info['player_state'][own_id]
        own_position= own_player_state.state.position

        door_state_dict={0:'closed',1:'open'}
        doors = self.info['doors']
        door_list = []
        
        # 预先将上千道门的位置信息进行储存，后续使用矩阵运算筛选出可见距离内的门！
        if self.doors_pos_matrix is None or len(self.features.doors_categorys)==0: 
            self.doors_pos_matrix = []
            for door_id, door in doors.items():
                self.features.doors_categorys.append(door_id)
                self.doors_pos_matrix.append([door.position.x,door.position.y,door.position.z])
            self.doors_pos_matrix = torch.as_tensor(self.doors_pos_matrix)
            self.features.doors_categorys = torch.as_tensor(self.features.doors_categorys)
        # 找出范围内的door
        visble_doors = self.find_nearest_doors([own_position.x,own_position.y,own_position.z])
        
        nearest_door_dist = np.inf
  
        for door_id in visble_doors:
            door_id = door_id.item()
            if door_id not in doors.keys():
                continue
            door = doors[door_id]
            # 绝对坐标
            door_pos = door.position
            # 相对坐标
            door_rel_x = own_position.x - door_pos.x
            door_rel_y = own_position.y - door_pos.y
            door_rel_z = own_position.z - door_pos.z
            # 与门之间的水平和垂直角度差值 
            # 1：开，0：关闭
            door_state = door_state_dict[door.state]
            # 距离门的距离
            distance = math.sqrt(door_rel_x ** 2 + door_rel_y ** 2 +
                                 door_rel_z ** 2)
 
            nearest_door_dist = min(nearest_door_dist,distance)
 
                         
            door_list.append([
                distance/self.position_norm,
                door_pos.x/self.position_norm,
                door_pos.y/self.position_norm,
                door_pos.z/self.position_norm,
                door_rel_x/self.position_norm,
                door_rel_y/self.position_norm,
                door_rel_z/self.position_norm,
                door_state, 
            ])
        sorted_door_list = sorted(door_list)
        number_of_visible_doors = len(sorted_door_list)
        if number_of_visible_doors==0:
            state_dict ={'number_of_visible_doors':number_of_visible_doors}
        else:
            nearest_door = sorted_door_list[0]
            distance = int(nearest_door[0])
            if distance<50:
                distance_str = f"{distance} (near)"
            elif distance<150:
                distance_str = f"{distance} (middle)"
            else:
                distance_str = f"{distance} (far)"            
            relative_direction = f"yaw_{int(self.calculate_move_angle(round(nearest_door[4],), round(nearest_door[5],)))}"   
            state_dict ={
                'number_of_visible_doors':number_of_visible_doors,
                'distance_to_me':distance_str,           
                'position':(round(nearest_door[1],),round(nearest_door[2],),round(nearest_door[3],)),
                'relative_position_to_me':(round(nearest_door[4],),round(nearest_door[5],),round(nearest_door[6],)),
                'door_state':door_state,  
                'which_direction_door_is_located_to_me':relative_direction,
            }
        # state_str_list = []
        # for k,v in state_dict.items():
        #     state_str_list.append(f"{k}:{v}")
        
        # state_str = '; '.join(state_str_list)
        # state_str = 'Near door states:'+state_str
        return state_dict       
    
    # def describe_backpack_state(self,features_info):        
    #     state_info = features_info['backpack_item_info'] 
    #     state_dict ={
    #         'main_type':state_info['main_type'].item(), 
    #         'count':state_info['count'].item(),             
    #         'slot_0':state_info['slot_0'].item(), 
    #         'slot_1':state_info['slot_1'].item(), 
    #     }
    #     state_str_list = []
    #     for k,v in state_dict.items():
    #         state_str_list.append(f"{k}:{v}")
        
    #     state_str = '; '.join(state_str_list)
    #     state_str = 'Backpack state are:'+state_str
    #     return state_str      

    def describe_event_state(self, ):        
        own_id = self.features.id    
        state_info = self.info['player_state']  
        own_player_state = state_info[own_id]
        own_position= own_player_state.state.position
        
        damage_source = own_player_state.damage.damage_source
 

        damage_list = []
        for damage in damage_source:
  
            if damage.damage_source_id >= 6000 and damage.damage_source_id < (6000+20):
                source_position = state_info[damage.damage_source_id].state.position

                damage2own_distance = self.computer_distance(source_position.x,source_position.y,source_position.z,own_position.x,own_position.y,own_position.z)
                damage_value = damage.damage
                damage_relative_me_pos_x = source_position.x - own_position.x
                damage_relative_me_pos_y = source_position.y - own_position.y
                damage_relative_me_pos_z = source_position.z - own_position.z
                damage_info = [damage2own_distance/self.position_norm,
                               damage_value,
                               source_position.x/self.position_norm,
                               source_position.y/self.position_norm,
                               source_position.z/self.position_norm,
                               damage_relative_me_pos_x/self.position_norm,
                               damage_relative_me_pos_y/self.position_norm,
                               damage_relative_me_pos_z/self.position_norm,
                               damage.damage_source_id
                               ]
                damage_list.append(damage_info)
        sorted_damage_list = sorted(damage_list)
               
        number_of_damage = len(sorted_damage_list)
        if number_of_damage==0:
            state_dict = {'damage':0}
        else:
            nearest_damage = sorted_damage_list[0]
            distance = nearest_damage[0]
            if distance<50:
                distance_str = f"{distance} (near)"
            elif distance<150:
                distance_str = f"{distance} (middle)"
            else:
                distance_str = f"{distance} (far)"            
            if self.nearest_enemy_id==nearest_damage[-1]:
                state_dict = {
                    'damage_from_nearest_enemy':'true',  
                    # 'distance_to_me':int(nearest_damage[0]),
                    'damage_value':int(nearest_damage[1]),                                  
                    # 'position':(round(nearest_damage[2],),round(nearest_damage[3],),round(nearest_damage[4],)),
                    # 'relative_position':(round(nearest_damage[5],),round(nearest_damage[6],),round(nearest_damage[7],)),
                    }
            else:
                relative_direction = f"yaw_{int(self.calculate_move_angle(round(nearest_damage[5],), round(nearest_damage[6],)))}"   
                state_dict = {
                'damage_from_nearest_enemy':'false',    
                'distance_to_me':distance_str,
                'damage_value':int(nearest_damage[1]),                              
                'damage_source_position':(round(nearest_damage[2],),round(nearest_damage[3],),round(nearest_damage[4],)),
                'relative_position_to_me':(round(nearest_damage[5],),round(nearest_damage[6],),round(nearest_damage[7],)),
                'which_direction_damage_is_located_relative_to_me': relative_direction,
                }
        # state_str_list = []
        # for k,v in state_dict.items():
        #     state_str_list.append(f"{k}:{v}")
        
        # state_str = '; '.join(state_str_list)
        # state_str = 'Damage states:'+state_str
        return state_dict



    def describe_step_state(self, iter_step=None,player_id=None,json_data=None,root_json_data=None): 
        traj_input = iter_step
        self.state = traj_input['state']
        self.objs = traj_input['objs']
        depth_map = traj_input['depth_map']
        last_state = traj_input['last_state']
        self.last_action = traj_input['last_action']
        self.info = traj_input['info']
        # self.search_target_player_id = self.find_search_target(self.info,player_id)
        if 'search_target_player_id' in traj_input.keys():
            self.search_target_player_id = traj_input['search_target_player_id']
        else:
            self.search_target_player_id = self.find_search_target(self.info,player_id)
        features_info = self.preprocess_obs_nonorm(state= self.state,
                                        objs= self.objs,
                                        depth_map= depth_map,
                                        last_state= last_state,
                                        last_action= self.last_action,
                                        info=self.info,
                                        search_target_player_id=self.search_target_player_id)
            
        # print(f"features_info:{features_info}")
        scalar_info = self.describe_myself_state(features_info)
        enemy_info = self.describe_enemy_state( ) 
        teammate_info = self.describe_teammate_state( )
        door_info = self.describe_door_state( )
        monster_info = self.describe_monster_state( )
        weapon_backpack_info = self.describe_weapon_backpack_state( )
        event_info = self.describe_event_state( )
        for w_k,w_v in weapon_backpack_info.items():
            scalar_info[w_k]=w_v
        # start_str_list = ['My states:','Nearest enemy states:','Nearest teammate states:','My suffered damage source:','Nearest door info:','Nearest monster states:']
        # full_descirbe_list = [scalar_info,enemy_info,teammate_info,event_info,door_info,monster_info]
        # descirbe_list = [] 
        # for idx,d in enumerate(full_descirbe_list):
        #     if len(d)>1:
        #         start_str = start_str_list[idx]
        #         state_str_list = []
        #         for k,v in d.items():
        #             state_str_list.append(f"{k}:{v}")
        #         state_str = '; '.join(state_str_list)
        #         state_str = start_str+state_str
        #         descirbe_list.append(state_str)
        # describe_state_str = '.\n'.join(descirbe_list)+'.'
        # print(describe_state_str)
        return scalar_info,enemy_info,teammate_info,door_info,monster_info,event_info



    def describe_state(self, trajectory_path=None,player_id=None,json_data=None,root_json_data=None): 
        # extends=json_data["Actions Extend Nums"]   
        trajectory = torch.load(trajectory_path)
        # if len(extends)==2:
        #     start_idx = max(extends[0],0)
        #     end_idx = min(len(trajectory)-extends[-1], len(trajectory))
        #     trajectory=trajectory[start_idx:end_idx]
 
        # random_idxs = list(range(len(trajectory)))
        # random.shuffle(random_idxs)
        for traj in trajectory:
            # traj = trajectory[i]
            self.src_action = traj[1]
            traj_input = traj[0]
            self.state = traj_input['state']
            self.objs = traj_input['objs']
            depth_map = traj_input['depth_map']
            last_state = traj_input['last_state']
            self.last_action = traj_input['last_action']
            self.info = traj_input['info']
            # self.search_target_player_id = self.find_search_target(self.info,player_id)
            if 'search_target_player_id' in traj_input.keys():
                self.search_target_player_id = traj_input['search_target_player_id']
            else:
                self.search_target_player_id = self.find_search_target(self.info,player_id)
            features_info = self.preprocess_obs_nonorm(state= self.state,
                                            objs= self.objs,
                                            depth_map= depth_map,
                                            last_state= last_state,
                                            last_action= self.last_action,
                                            info=self.info,
                                            search_target_player_id=self.search_target_player_id)
            
            # print(f"features_info:{features_info}")
            scalar_info = self.describe_myself_state(features_info)
            enemy_info = self.describe_enemy_state( ) 
            teammate_info = self.describe_teammate_state( )
            door_info = self.describe_door_state( )
            monster_info = self.describe_monster_state( )
            weapon_backpack_info = self.describe_weapon_backpack_state( )
            event_info = self.describe_event_state( )
            for w_k,w_v in weapon_backpack_info.items():
                scalar_info[w_k]=w_v
            start_str_list = ['My states:','Nearest enemy states:','Nearest teammate states:','My suffered damage source:','Nearest door info:','Nearest monster states:']
            full_descirbe_list = [scalar_info,enemy_info,teammate_info,event_info,door_info,monster_info]
            descirbe_list = [] 
            for idx,d in enumerate(full_descirbe_list):
                if len(d)>1:
                    start_str = start_str_list[idx]
                    state_str_list = []
                    for k,v in d.items():
                        state_str_list.append(f"{k}:{v}")
                    state_str = '; '.join(state_str_list)
                    state_str = start_str+state_str
                    descirbe_list.append(state_str)
            describe_state_str = '.\n'.join(descirbe_list)+'.'
            print(describe_state_str)
        return scalar_info,enemy_info,teammate_info,door_info,monster_info,event_info
            
    def get_backbone_embedding_onestep(self, model_input ):
        model_output = self.model.get_backbone_embedding(model_input)
        embed = model_output['embedding'].cpu().data.numpy()
        return embed    
    def get_backbone_embedding_expand_onestep(self, model_input ):
        model_output = self.model.get_backbone_embedding_expand(model_input)
        embed = model_output['embedding'].cpu().data.numpy()
        return embed       
    
    def get_action(self, model_input ):
        self.model_output = self.model.compute_logp_action(model_input)
        # embed = model_output['embedding'].cpu().data.numpy()
        actions = self.features.transform_actions(self.model_output['action'],state=self.state,info=self.info,objs=self.objs,target_unit=self.model_output['action']["target_unit"])
        print(f"cur model out action:{self.model_output['action']}")
        print(f"src model out action:{self.src_action['model_output_action']}")
        print(f"cur actions:{actions[0]}")
        print(f"src actions:{self.src_action['actions']}")
        action_idx = self.src_action['model_output_action']['single_head'].item()
        action_str = f"{self.output2action_dict_old[action_idx][0]}"+'_'+f"{self.output2action_dict_old[action_idx][1]}"
        print(action_str)

        return actions    

def load_json(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data
 

def get_traj_embedding(config_path=None):
    traj_agent = Trajectory(config_path=config_path)
    # root_clips_path = '/mnt/nfs2/aaa/FPS/Label_3player'
    root_clips_path = '/mnt/nfs2/aaa/FPS_Data_4P_Trajs'
    clip_replay_names = os.listdir(root_clips_path)
 
    for idx,replay in enumerate(clip_replay_names):
        # replay = 'server12-6003-1109-2024-03-25-16-34-57.pt'
        
        print(f"正在处理第{idx}个视频{replay}的clip...")
        clip_path = os.path.join(root_clips_path, replay)
        # root_clip_files = os.listdir(clip_path)
 
        if '.pt' in clip_path:
            
            player_id = int(replay.split('-')[1])
            traj_path = clip_path
            
            
            # model_input = traj_agent.load_trajectory_onestep(traj_path,json_data, root_json_data )
            traj_agent.features.id = player_id
            traj_agent.features_nonorm.id = player_id
            model_input = traj_agent.load_trajectory_onestep(traj_path )
            scalar_info,enemy_info,teammate_info,door_info,monster_info,event_info \
                = traj_agent.describe_state(traj_path,player_id)
            
            embedding  = traj_agent.get_backbone_embedding_onestep(model_input)
            # action  = traj_agent.get_action(model_input)
            # print(f"embedding:{action}")
def get_parts_embedding(config_path=None, sample_freq=10):
    traj_agent = Trajectory(config_path=config_path)
    # root_clips_path = '/mnt/nfs2/aaa/FPS/Label_3player'
    root_parts_path = '/mnt/nfs2/aaa/FPS'
    part_names = ['4P_Align_Part1','4P_Align_Part2','4P_Align_Part3','4P_Align_Part4','4P_Align_Part5']
    
    for part in part_names:
        part_path = os.path.join(root_parts_path, part)
        replay_names = os.listdir(part_path)
        for idx,replay in enumerate(replay_names):
            # replay = 'server12-6003-1109-2024-03-25-16-34-57.pt'
            
            print(f"正在处理第{idx}个视频{replay}的clip...")
            json_path = os.path.join(part_path, replay, f"{replay}.json")
            json_data = load_json(json_path)
            traj_length = len(json_data)
            for sub_dict in json_data[::sample_freq]:
                state = sub_dict['state']
                state_sub_path,player_id = state.split(":")[0],state.split(":")[1]
                state_path = os.path.join(part_path,state_sub_path)
         
                
                
                # model_input = traj_agent.load_trajectory_onestep(traj_path,json_data, root_json_data )
                traj_agent.features.id = player_id
                # traj_agent.features_nonorm.id = player_id
                model_input = traj_agent.load_state_onestep(state_path )
                # scalar_info,enemy_info,teammate_info,door_info,monster_info,event_info \
                #     = traj_agent.describe_state(traj_path,player_id)
                
                embedding  = traj_agent.get_backbone_embedding_onestep(model_input)
                # action  = traj_agent.get_action(model_input)
                # print(f"embedding:{action}")            
  
# get_traj_embedding(config_path='./utils/user_config_replay.yaml')
# get_parts_embedding(config_path='./utils/user_config_replay.yaml')
