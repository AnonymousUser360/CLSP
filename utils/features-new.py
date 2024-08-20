import math
import random

import numpy as np
import torch
from collections import deque
import copy
from typing import List
import torch.nn as nn
from bigrl.core.utils.heatmap_utils.heatmap_helper import HeatMap
import time

from fps.proto.ccs_ai import Action, AIAction, AIStateResponse, PlayerResultInfo, Vector2, Vector3, \
    ActionFocus, ActionMove, ActionRun, ActionSlide, ActionCrouch, ActionDrive, ActionJump, \
    ActionGround, ActionFire, ActionAim, ActionSwitchWeapon, ActionReload, ActionPick, ActionConsumeItem, \
    ActionDropItem, ActionHoldThrownItem, \
    ActionCancelThrow, ActionThrow, ActionRescue, ActionDoorSwitch, ActionSwim, ActionOpenAirDrop, ActionType

from .env_info.supply_item_info import all_supply_items
from .env_info.character_info import all_characters
from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.core.utils.config_helper import deep_merge_dicts
from .model.model import default_config as default_model_config
from .action_utils import pick, drop, move, target_move, action_response, jump, switch_door
import pandas as pd
from .supply.supply_moule_item_info import all_supply_items as ALL_SUPPLY_ITEMS
from .supply.supply_moule_item_info import AttachmentUseble,AttachmentCategorys,GunCategorys,bullet2gun,gun2bullet,AttachmentsLocs,RecoveryHP,MONSTER
from collections import defaultdict
from collections import deque
from .supply_scripts import SupplyModule

WEAPON_DISTANCE = {2101: 50000, 2102: 50000, 2103: 50000, 2104: 50000, 2201: 70000, 2202: 70000, 2203: 70000,
                   2401: 30000, 2402: 30000, 2403: 30000, 2407: 30000,
                   2501: 8000, 2502: 8000, 2601: 50000, 2602: 50000, 2801: 20000, 2802: 150000, 2803: 100000,
                   2804: 8000}
DOOR_CAN_OPEN_CLOSE_DISTANCE = 300
TEAMMATE_CAN_BE_SAVED_DISTANCE = 300 
RECOVER_ITEM = {3301: 25, 3302: 100}

class Features:
    def __init__(self, cfg , heat_map_data = None):

        self.heatmap_tool = HeatMap(cfg=cfg,heat_map_data=heat_map_data)
        self.cfg = deep_merge_dicts(default_model_config, cfg)
        self.agent_idx = self.cfg.agent.get('agent_idx', 0)
        self.id = 6000 + self.agent_idx
        self.spatial_size = reversed(self.cfg.game.depthmap.resolution) # reversed([64,32])
        self.MAX_DEPTH = 100
        self.MAX_POS = 100
        self.MAX_HP = 100

        self.MOVE_DIR_LIST = self.cfg.agent.actions.get('move_dir',
                                                        [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195,
                                                         210, 225, 240, 255, 270, 285, 300, 315, 330, 345])
        self.YAW_LIST = self.cfg.agent.actions.get('yaw', [0, 1, 2, 5, 10, 20, -1, -2, -5, -10, -20, ])
        self.PITCH_LIST = self.cfg.agent.actions.get('pitch', [0, 0.5, 1, 2, 5, 10, 25, -0.5, -1, -2, -5, -10, -25, ])
        self.BODY_ACTION_LIST = self.cfg.agent.actions.get('body_action', ["none", "stop", "run", "slide" , "crouch", "jump", "ground"])

        # self.JUMP_LIST = self.cfg.agent.actions.jump

        self.team_num = self.cfg.env.team_num
        self.player_num_per_team = self.cfg.env.player_num_per_team
        self.max_player_num = self.team_num * self.player_num_per_team
        self.max_visible_player_num = 10
        self.max_enemy_num = 20
        self.max_supply_item_num = self.cfg.agent.get('max_supply_item_num', 20)
        self.max_backpack_item_num = self.cfg.agent.get('max_backpack_item_num', 20)
        self.max_player_weapon_num = self.cfg.agent.get('max_player_weapon_num', 5)
        self.max_door_num = self.cfg.agent.get('max_door_num', 20)
        self.max_monster_num = self.cfg.agent.get('max_monster_num', 20)
        self.rel_move_dir = self.cfg.agent.get('rel_move_dir', False)
        self.rel_move_angle = self.cfg.agent.get('rel_move_angle', True)
        
        self.max_slot_nums = self.cfg.agent.get('max_slot_nums', 5)
        self.max_attachment_nums = self.cfg.agent.get('max_attachment_nums', 5)
        self.doors_visble_distance = self.cfg.agent.get('doors_visble_distance', 3000)
        self.stay_at_supply_maxstep = self.cfg.agent.get('stay_at_supply_maxstep', 150)

        self.Max_Init_Enemy_Num = 20
        self.Max_Init_Supply_Num = 40
        self.MAX_Event_Num = 20
        self.MAX_ROTATION_Num = 50
        self.teammate_num = 5
        
        self.heat_map_scale = self.cfg.env.heat_map.get('heat_map_scale', 10)
        
        
        self.heat_map_max_teammate_nums = self.cfg.env.heat_map.get('heat_map_max_teammate_nums', 5)
        self.heat_map_max_enemy_nums = self.cfg.env.heat_map.get('heat_map_max_enemy_nums', 20)
        self.heat_map_max_door_nums = self.cfg.env.heat_map.get('heat_map_max_enemy_nums', 10)
        self.heat_map_deltas = self.cfg.env.heat_map.get('heat_map_deltas', {})
        
        heat_map_size = self.cfg.env.heat_map.get('heat_map_size', [300,300])
        self.heat_map_model_size = [heat_map_size[0],heat_map_size[1]]
        
        
        self.target_supply_mode = self.cfg.agent.get('target_supply_mode', "pritory")
        
        # self.heat_map_center = [(local_heat_map_size[0] )//2,
        #                (local_heat_map_size[1] )//2]
        self.all_pool_map_size = {}
        self.kernel_size = 0
        for k,v in self.heat_map_deltas.items():
            if k not in  ["bottom2top_10","pitch_delta_10"]:
                for delta in v:
                    self.kernel_size += 1
                    if delta not in self.all_pool_map_size.keys():
                        self.all_pool_map_size[delta] = [(delta * heat_map_size[0])//2,
                                                        (delta * heat_map_size[1])//2]
        # self.kernel_size += 1
        self.setup()
        self._init_fake_data()
        
    def setup(self):
        # =============================================
        # 生成动作与模型输出的匹配信息和初始化action mask @wj
        # =============================================
        self.heatmap_enemy_position = {}
        self.heatmap_enemy_time_delta = 10
        self.ideal_speed_dir = [None, None,None,None,None]
        # self.stuck_buffer = deque(maxlen=5)
        # for _ in range(5):
        #     self.stuck_buffer.append(0)
        
        self.history_event_info = deque(maxlen=self.MAX_Event_Num)
        self.last_hp = None
        self.last_neardeath_hp = None
        self.history_supply_target_id = []
        self.agent_static_info = {
            "approch_target_cnt":0
        }
        
        self.reward_info = {}
        
        self.nearest_supply = None
        
        self.backpack_type_buffers = []
        self.supply_script = SupplyModule()
        self.supply_script.reset()
        action_list = self.cfg.agent.action_heads
        
        self.approch_supply_step = 0

        self.model_output2action = {}
        self.action_mask = {}
        self.action2model_output = {}
        
        for head in action_list:
            idx = 0
            self.model_output2action[head] = {}
            self.action_mask[head] = []
            for head_sub in self.cfg.agent.action_head_sub[head]:
                self.action2model_output[head_sub] = {}
                for val in self.cfg.agent.actions_range[head_sub]:
                    self.model_output2action[head][idx] = (head_sub,val)
                    self.action_mask[head].append(0)
                    self.action2model_output[head_sub][val] = idx
                    idx += 1
                         
        self.MONSTER = MONSTER()

        self.game_start_time = None
        self.ray_dirs_norm = 1000
        self.teammate_info_size = 75
        self.rotation_norm = 180
        self.rotation_time_norm = 120000
        self.size_xy_norm = 50
        self.size_z_norm = 200
        self.speed_norm = 750
        self.expect_speed = 500
        self.hp_norm = 100
        self.monster_hp_norm = 3000
        self.oxygen_norm = 100
        self.peek_norm = 3
        self.alive_norm = 3
        self.body_norm = 8
        self.backup_volume_norm = 250
        self.bullet_norm = 100
        self.reloading_time_norm = 10000 
        self.mapsize = [403300,403300,33000]
        self.camera_position_norm = 200
        self.quantity_norm = 300
        
        self.max_player_num_in_team = 5
        self.teammate_dict = None

        self.MAX_GAME_TIME = 2400
        self.norm_time_visble_enemy = 30
        self.supply_visble_distance = 3000
        self.pain_norm = 10
        
        self.ALL_SUPPLY_ITEMS = ALL_SUPPLY_ITEMS
        self.AttachmentUseble = AttachmentUseble
        self.AttachmentCategorys = AttachmentCategorys
        self.GunCategorys = GunCategorys
        self.bullet2gun = bullet2gun
        self.gun2bullet = gun2bullet
        self.AttachmentsLocs = AttachmentsLocs
        self.doors_pos_matrix = None
        self.doors_categorys = []
        
        self.monsters_visble_distance = 10000
        


        self.bag_volumes = {"default":100,
                                3221:150,
                                3222:200,
                                3223:250,
                                }

        self.fire_no_stop_distance = 5000

        self.supplys_pos_matrix = None
        self.supplys_categorys = []
        self.ENEMY_SEE_INFO = {}
        self.SUPPLY_SEE_INFO = {}
        self.enemy_id_queue = deque(maxlen=self.Max_Init_Enemy_Num)
        self.enemy_info_queue = deque(maxlen=self.Max_Init_Enemy_Num)
        self.supply_id_queue = deque(maxlen=self.Max_Init_Supply_Num)
        self.supply_info_queue = deque(maxlen=self.Max_Init_Supply_Num)
        self.max_supply_num = 50
        self.v_inspect_enemy_distance = 60000
        
        self.low_level_hp = 20
        self.middle_level_hp = 75
        self.fire_pitch_threshold = 1
        self.fire_yaw_threshold = 5
        self.open_aim_percent = 0.02
        
        # 加入自己的历史轨迹信息
        self.self_history_positions = deque(maxlen=600)
        for _ in range(600):
            self.self_history_positions.append([-1,-1])
        
        self.position_history_lens = 60
        
        
        
        # 初始化背包物品
        self.backpack_update_info = {
            "item":[],
            "priority":[],
            "count":[],
        }
        
        for item in self.ALL_SUPPLY_ITEMS.keys():
            self.backpack_update_info["item"].append(item)
            self.backpack_update_info["priority"].append(self.ALL_SUPPLY_ITEMS[item].priority)
            self.backpack_update_info["count"].append(0)
        
        # self.nearest_supply = None

        self.search_time_delta = self.cfg.agent.reward.get('search_time_delta', 10)

        self.all_player_nums = None
        self.alive_player_nums = None
        self.all_teammate_nums = self.player_num_per_team
        self.alive_teammate_nums = None
        # 过去的rotation
        self.history_rotation = deque(maxlen=1)
        self.rotation_info_all_list = deque(maxlen=self.MAX_ROTATION_Num)
        self.rotation_change_time = 0
            
        
        
    def init_last_action(self):
        pass

    def init_info(self):
        info = {}
        return info

    def _init_fake_data(self):
        
        
        self.EVENT_INFO = {
            
            'main_type':  (torch.long, (self.MAX_Event_Num,)),
            'sub_type': (torch.long, (self.MAX_Event_Num,)),
            'x': (torch.float, (self.MAX_Event_Num,)),
            'y': (torch.float, (self.MAX_Event_Num,)),
            'z': (torch.float, (self.MAX_Event_Num,)),     
            'damage': (torch.float, (self.MAX_Event_Num,)),
            'time_delta':(torch.float, (self.MAX_Event_Num,)),
            'tmp_1': (torch.float, (self.MAX_Event_Num,)),
            'tmp_2': (torch.float, (self.MAX_Event_Num,)),
            'tmp_3': (torch.float, (self.MAX_Event_Num,)),
            'tmp_4': (torch.float, (self.MAX_Event_Num,)),
            'event_num': (torch.long, ()),
            
           
        }

        self.ROTATION_INFO = {            
            'x': (torch.float, (self.MAX_ROTATION_Num,)),
            'y': (torch.float, (self.MAX_ROTATION_Num,)),
            'z': (torch.float, (self.MAX_ROTATION_Num,)),     
            'rotation_x': (torch.float, (self.MAX_ROTATION_Num,)),
            'rotation_y': (torch.float, (self.MAX_ROTATION_Num,)),
            'rotation_z': (torch.float, (self.MAX_ROTATION_Num,)),   
            'history_rotation_x': (torch.float, (self.MAX_ROTATION_Num,)),
            'history_rotation_y': (torch.float, (self.MAX_ROTATION_Num,)),
            'history_rotation_z': (torch.float, (self.MAX_ROTATION_Num,)),   
            'delta_rotation_x': (torch.float, (self.MAX_ROTATION_Num,)),
            'delta_rotation_y': (torch.float, (self.MAX_ROTATION_Num,)),
            'delta_rotation_z': (torch.float, (self.MAX_ROTATION_Num,)),               
            'time':(torch.float, (self.MAX_ROTATION_Num,)),
            'see_enemy': (torch.long, (self.MAX_ROTATION_Num,)),
            "delta_pos_x":(torch.float, (self.MAX_ROTATION_Num,)),
            "delta_pos_y":(torch.float, (self.MAX_ROTATION_Num,)),
            "delta_pos_z":(torch.float, (self.MAX_ROTATION_Num,)),
            "distance":(torch.float, (self.MAX_ROTATION_Num,)),
            "current_delta_rotation_x":(torch.float, (self.MAX_ROTATION_Num,)),
            "current_delta_rotation_y":(torch.float, (self.MAX_ROTATION_Num,)),
            "current_delta_rotation_z":(torch.float, (self.MAX_ROTATION_Num,)),
            'rotation_num': (torch.long, ()),
            
        }
 
        
        self.HISTORY_POSITIONS_INFO = (torch.float, (self.position_history_lens * 2,))
        
        self.MASK_INFO = {
            k:(torch.long, (len(v),)) for k,v in self.action_mask.items()
        }
        
        self.ACTION_INFO = {
            k:(torch.long, ()) for k in self.action_mask.keys()
        }
        self.ACTION_INFO["target_unit"] = (torch.long, ())
        self.ACTION_LOGP = {
            k:(torch.float, ()) for k in self.action_mask.keys()
        }

        self.ACTION_LOGP["target_unit"] = (torch.float, ())

        self.SCALAR_INFO = {
            'backpack_volume_total': (torch.float, ()),           
            'backpack_volume_rest': (torch.float, ()),           
            'backpack_volume_percent': (torch.float, ()),
            "is_treat": (torch.long, ()),
            "treat_remain_time": (torch.float, ()),
            "is_rescue": (torch.long, ()),
            "rescue_remain_time": (torch.float, ()),
            "is_rescued": (torch.long, ()),
            "rescued_remain_time": (torch.float, ()),
            "is_reloading": (torch.long, ()),
            "reloading_remain_time": (torch.float, ()),
            "own_safety_area_state": (torch.long, ()),
            "own_safety_area_pos_x": (torch.float, ()),
            "own_safety_area_pos_y": (torch.float, ()),
            "own_safety_area_radius": (torch.float, ()),
            "own_safety_area_next_pos_x": (torch.float, ()),
            "own_safety_area_next_pos_y": (torch.float, ()),
            "own_safety_area_next_radius": (torch.float, ()),
            "safety_area_time": (torch.float, ()),
            "safety_area_total_time": (torch.float, ()),
            "own_safety_area_rest_time": (torch.float, ()),
            "own_player_in_blue_safetyarea": (torch.long, ()),
            "own_player_in_white_safetyarea": (torch.long, ()),
            "own_player_vec_blue_safetyarea_x": (torch.float, ()),
            "own_player_vec_blue_safetyarea_y": (torch.float, ()),
            "own_player_vec_white_safetyarea_x": (torch.float, ()),
            "own_player_vec_white_safetyarea_y": (torch.float, ()),
            "own_dis_blue_safetyarea": (torch.float, ()),
            "own_dis_white_safetyarea": (torch.float, ()),
            "own_dis_blue_safetyarea_radius": (torch.float, ()),
            "own_dis_white_safetyarea_radius": (torch.float, ()),
            "own_whether_run_in_blue_circle_time": (torch.float, ()),
            "own_whether_run_in_blue_circle": (torch.float, ()),
            "own_whether_run_in_white_circle": (torch.float, ()),
            "safezone_pain": (torch.float, ()),
            "safezone_appear_time": (torch.float, ()),
            "safezone_delay_time": (torch.float, ()),
            
            'character_id': (torch.long, ()),
            'team_id' : (torch.long, ()),
            'position_x': (torch.float, ()),
            'position_y': (torch.float, ()),
            'position_z': (torch.float, ()),

            #TODO
            'rotation_x': (torch.float, ()),
            'rotation_y': (torch.float, ()),
            'rotation_z': (torch.float, ()),
            'sin_rotation_x': (torch.float, ()),
            'cos_rotation_x': (torch.float, ()),
            'sin_rotation_y': (torch.float, ()),
            'cos_rotation_y': (torch.float, ()),
            'sin_rotation_z': (torch.float, ()),
            'cos_rotation_z': (torch.float, ()),

            'size_x': (torch.float, ()),
            'size_y': (torch.float, ()),
            'size_z': (torch.float, ()),

            'speed_x': (torch.float, ()),
            'speed_y': (torch.float, ()),
            'speed_z': (torch.float, ()),
            'speed_scalar': (torch.float, ()),

            'hp': (torch.float, ()),
            # 'hp_delta': (torch.float, ()),
            'neardeath_breath': (torch.float, ()),
            'oxygen': (torch.float, ()),
            # 'buff': (torch.long, ()),
            'peek_type': (torch.long, ()),
            'alive_state': (torch.long, ()),
            'body_state': (torch.long, ()),

            'is_switching': (torch.long, ()),
            'is_pose_changing': (torch.long, ()),
            'is_running': (torch.long, ()),
            'is_aiming': (torch.long, ()),
            'is_firing': (torch.long, ()),
            'is_holding': (torch.long, ()),
            'is_falling': (torch.long, ()),
            'is_picking': (torch.long, ()),

            'camera_x': (torch.float, ()),
            'camera_y': (torch.float, ()),
            'camera_z': (torch.float, ()),

            'camera_rotation_x': (torch.float, ()),
            'camera_rotation_y': (torch.float, ()),
            'camera_rotation_z': (torch.float, ()),
            
            'sin_camera_rotation_x': (torch.float, ()),
            'cos_camera_rotation_x': (torch.float, ()),
            'sin_camera_rotation_y': (torch.float, ()),
            'cos_camera_rotation_y': (torch.float, ()),
            'sin_camera_rotation_z': (torch.float, ()),
            'cos_camera_rotation_z': (torch.float, ()),
            'skill_buff_1': (torch.long, ()),
            'skill_buff_2': (torch.long, ()),
            'skill_buff_3': (torch.long, ()),
            "last_action": (torch.long, ()),

            'target_x': (torch.float, ()),
            'target_y': (torch.float, ()),
            'target_z': (torch.float, ()),
            'target_x_rel': (torch.float, ()),
            'target_y_rel': (torch.float, ()),
            'target_z_rel': (torch.float, ()),
            'target_distance': (torch.float, ()),
            "not_visble_enemy_time": (torch.float, ()),

            "all_player_nums": (torch.float, ()),
            "alive_player_nums": (torch.float, ()),
            "player_alive2all_ratio": (torch.float, ()),
            "all_teammate_nums": (torch.float, ()),
            "alive_teammate_nums": (torch.float, ()),
            "teammate_alive2all_ratio": (torch.float, ()),

            'dir_distance_1':(torch.float, ()),
            'dir_distance_2':(torch.float, ()),
            'dir_distance_3':(torch.float, ()),
            'dir_distance_4':(torch.float, ()),
            'dir_distance_5':(torch.float, ()),
            'dir_distance_6':(torch.float, ()),
            'dir_distance_7':(torch.float, ()),
            'dir_distance_8':(torch.float, ()),
            'dir_distance_9':(torch.float, ()),
            'dir_distance_10':(torch.float, ()),
            'dir_distance_11':(torch.float, ()),
            'dir_distance_12':(torch.float, ()),
            'dir_distance_13':(torch.float, ()),
            'dir_distance_14':(torch.float, ()),
            'dir_distance_15':(torch.float, ()),
            'dir_distance_16':(torch.float, ()),
        }



        self.TEAMMATE_INFO = {
            "character": (torch.long, (self.teammate_num,)),
            'teammate_team_id': (torch.long, (self.teammate_num,)),
            'teammate_pos_x': (torch.float, (self.teammate_num,)),
            'teammate_pos_y': (torch.float, (self.teammate_num,)),
            'teammate_pos_z': (torch.float, (self.teammate_num,)),
            'teammate_rotation_x': (torch.float, (self.teammate_num,)),
            'teammate_rotation_y': (torch.float, (self.teammate_num,)),
            'teammate_rotation_z': (torch.float, (self.teammate_num,)),
            'teammate_size_x': (torch.float, (self.teammate_num,)),
            'teammate_size_y': (torch.float, (self.teammate_num,)),
            'teammate_size_z': (torch.float, (self.teammate_num,)),
            'teammate_speed_x': (torch.float, (self.teammate_num,)),
            'teammate_speed_y': (torch.float, (self.teammate_num,)),
            'teammate_speed_z': (torch.float, (self.teammate_num,)),
            'teammate_scalar_speed': (torch.float, (self.teammate_num,)),
            'teammate_hp': (torch.float, (self.teammate_num,)),
            'teammate_neardeath_breath': (torch.float, (self.teammate_num,)),
            'teammate_oxygen': (torch.float, (self.teammate_num,)),
            'teammate_peek': (torch.long, (self.teammate_num,)),
            'teammate_alive': (torch.long, (self.teammate_num,)),
            'teammate_bodystate': (torch.long, (self.teammate_num,)),
            'teammate_camera_position_x': (torch.float, (self.teammate_num,)),
            'teammate_camera_position_y': (torch.float, (self.teammate_num,)),
            'teammate_camera_position_z': (torch.float, (self.teammate_num,)),
            'teammate_camera_rotation_x': (torch.float, (self.teammate_num,)),
            'teammate_camera_rotation_y': (torch.float, (self.teammate_num,)),
            'teammate_camera_rotation_z': (torch.float, (self.teammate_num,)),
            'teammate_sin_rotation_x': (torch.float, (self.teammate_num,)),
            'teammate_cos_rotation_x': (torch.float, (self.teammate_num,)),
            'teammate_sin_rotation_y': (torch.float, (self.teammate_num,)),
            'teammate_cos_rotation_y': (torch.float, (self.teammate_num,)),
            'teammate_sin_rotation_z': (torch.float, (self.teammate_num,)),
            'teammate_cos_rotation_z': (torch.float, (self.teammate_num,)),
            'teammate_is_switching': (torch.long, (self.teammate_num,)),
            'teammate_is_pose_changing': (torch.long, (self.teammate_num,)),
            'teammate_is_running': (torch.long, (self.teammate_num,)),
            'teammate_is_aiming': (torch.long, (self.teammate_num,)),
            'teammate_is_firing': (torch.long, (self.teammate_num,)),
            'teammate_is_holding': (torch.long, (self.teammate_num,)),
            'teammate_is_falling': (torch.long, (self.teammate_num,)),
            'teammate_is_picking': (torch.long, (self.teammate_num,)),
            'teammate_player_vec_x': (torch.float, (self.teammate_num,)),
            'teammate_player_vec_y': (torch.float, (self.teammate_num,)),
            'teammate_player_vec_z': (torch.float, (self.teammate_num,)),
            'teammate_player_dis': (torch.float, (self.teammate_num,)),
            'teammate_can_see_me': (torch.long, (self.teammate_num,)),
            'teammate_sin_camera_rotation_x': (torch.float, (self.teammate_num,)),
            'teammate_cos_camera_rotation_x': (torch.float, (self.teammate_num,)),
            'teammate_sin_camera_rotation_y': (torch.float, (self.teammate_num,)),
            'teammate_cos_camera_rotation_y': (torch.float, (self.teammate_num,)),
            'teammate_sin_camera_rotation_z': (torch.float, (self.teammate_num,)),
            'teammate_cos_camera_rotation_z': (torch.float, (self.teammate_num,)),
            'teammate_in_blue_safetyarea': (torch.long, (self.teammate_num,)),
            'teammate_in_white_safetyarea': (torch.long, (self.teammate_num,)),
            'teammate_vec_blue_safetyarea_x': (torch.float, (self.teammate_num,)),
            'teammate_vec_blue_safetyarea_y': (torch.float, (self.teammate_num,)),
            'teammate_vec_white_safetyarea_x': (torch.float, (self.teammate_num,)),
            'teammate_vec_white_safetyarea_y': (torch.float, (self.teammate_num,)),
            'teammate_dis_blue_safetyarea_map': (torch.float, (self.teammate_num,)),
            'teammate_dis_white_safetyarea_map': (torch.float, (self.teammate_num,)),
            'teammate_dis_blue_safetyarea_radius': (torch.float, (self.teammate_num,)),
            'teammate_dis_white_safetyarea_radius': (torch.float, (self.teammate_num,)),
            'whether_teammate_run_in_circle_time':(torch.float, (self.teammate_num,)),
            'whether_teammate_run_in_blue_circle': (torch.float, (self.teammate_num,)),
            'whether_teammate_run_in_white_circle':(torch.float, (self.teammate_num,)),
            'teammate_buff_1': (torch.long, (self.teammate_num,)),
            'teammate_buff_2': (torch.long, (self.teammate_num,)),
            'teammate_buff_3': (torch.long, (self.teammate_num,)),          
        }



        self.ENEMY_INFO = {
            'distance': (torch.float, (self.Max_Init_Enemy_Num,)),
            'team_id': (torch.long, (self.Max_Init_Enemy_Num,)),
            'pos_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'pos_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'pos_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_sin_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_sin_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_sin_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_cos_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_cos_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_cos_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'size_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'size_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'size_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'speed_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'speed_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'speed_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'scalar_speed': (torch.float, (self.Max_Init_Enemy_Num,)),
            'hp': (torch.float, (self.Max_Init_Enemy_Num,)),
            'neardeath_breath': (torch.float, (self.Max_Init_Enemy_Num,)),
            'oxygen': (torch.float, (self.Max_Init_Enemy_Num,)),
            'peek': (torch.long, (self.Max_Init_Enemy_Num,)),
            'alive': (torch.long, (self.Max_Init_Enemy_Num,)),
            'bodystate': (torch.long, (self.Max_Init_Enemy_Num,)),
            'relative_pos_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'relative_pos_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'relative_pos_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'character': (torch.long, (self.Max_Init_Enemy_Num,)),          
            'enemy_see_me': (torch.long, (self.Max_Init_Enemy_Num,)),           
            'enemy_relative_blue_safetyarea_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_relative_blue_safetyarea_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_relative_white_safetyarea_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_relative_white_safetyarea_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_distance_blue_safetyarea': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_distance_blue_safetyarea_relative': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_distance_white_safetyarea': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_distance_white_safetyarea_relative': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_in_blue_safetyarea': (torch.long, (self.Max_Init_Enemy_Num,)),
            'enemy_in_white_safetyarea': (torch.long, (self.Max_Init_Enemy_Num,)),
            'whether_enemy_run_in_blue_circle_time': (torch.float, (self.Max_Init_Enemy_Num,)),
            'whether_enemy_run_in_blue_circle': (torch.float, (self.Max_Init_Enemy_Num,)),
            'whether_enemy_run_in_white_circle': (torch.float, (self.Max_Init_Enemy_Num,)),
            'since_last_see_time': (torch.float, (self.Max_Init_Enemy_Num,)),
            "hold_gun" : (torch.long, (self.Max_Init_Enemy_Num,)),
            'enemy_buff_1': (torch.long, (self.Max_Init_Enemy_Num,)),
            'enemy_buff_2': (torch.long, (self.Max_Init_Enemy_Num,)),
            'enemy_buff_3': (torch.long, (self.Max_Init_Enemy_Num,)),           
            'enemy_item_num': (torch.long, ()),
            
        }


        self.VISIBLE_ENEMY_INFO = {
            'distance': (torch.float, (self.max_visible_player_num,)),
            'team_id': (torch.long, (self.max_visible_player_num,)),
            'pos_x': (torch.float, (self.max_visible_player_num,)),
            'pos_y': (torch.float, (self.max_visible_player_num,)),
            'pos_z': (torch.float, (self.max_visible_player_num,)),
            'rotation_x': (torch.float, (self.max_visible_player_num,)),
            'rotation_y': (torch.float, (self.max_visible_player_num,)),
            'rotation_z': (torch.float, (self.max_visible_player_num,)),
            'rotation_sin_x': (torch.float, (self.max_visible_player_num,)),
            'rotation_sin_y': (torch.float, (self.max_visible_player_num,)),
            'rotation_sin_z': (torch.float, (self.max_visible_player_num,)),
            'rotation_cos_x': (torch.float, (self.max_visible_player_num,)),
            'rotation_cos_y': (torch.float, (self.max_visible_player_num,)),
            'rotation_cos_z': (torch.float, (self.max_visible_player_num,)),
            'size_x': (torch.float, (self.max_visible_player_num,)),
            'size_y': (torch.float, (self.max_visible_player_num,)),
            'size_z': (torch.float, (self.max_visible_player_num,)),
            'speed_x': (torch.float, (self.max_visible_player_num,)),
            'speed_y': (torch.float, (self.max_visible_player_num,)),
            'speed_z': (torch.float, (self.max_visible_player_num,)),
            'scalar_speed': (torch.float, (self.max_visible_player_num,)),
            'hp': (torch.float, (self.max_visible_player_num,)),
            'neardeath_breath': (torch.float, (self.max_visible_player_num,)),
            'oxygen': (torch.float, (self.max_visible_player_num,)),
            'peek': (torch.long, (self.max_visible_player_num,)),
            'alive': (torch.long, (self.max_visible_player_num,)),
            'bodystate': (torch.long, (self.max_visible_player_num,)),
            'relative_pos_x': (torch.float, (self.max_visible_player_num,)),
            'relative_pos_y': (torch.float, (self.max_visible_player_num,)),
            'relative_pos_z': (torch.float, (self.max_visible_player_num,)),
            'character': (torch.long, (self.max_visible_player_num,)),          
            'enemy_see_me': (torch.long, (self.max_visible_player_num,)),           
            'enemy_relative_blue_safetyarea_x': (torch.float, (self.max_visible_player_num,)),
            'enemy_relative_blue_safetyarea_y': (torch.float, (self.max_visible_player_num,)),
            'enemy_relative_white_safetyarea_x': (torch.float, (self.max_visible_player_num,)),
            'enemy_relative_white_safetyarea_y': (torch.float, (self.max_visible_player_num,)),
            'enemy_distance_blue_safetyarea': (torch.float, (self.max_visible_player_num,)),
            'enemy_distance_blue_safetyarea_relative': (torch.float, (self.max_visible_player_num,)),
            'enemy_distance_white_safetyarea': (torch.float, (self.max_visible_player_num,)),
            'enemy_distance_white_safetyarea_relative': (torch.float, (self.max_visible_player_num,)),
            'enemy_in_blue_safetyarea': (torch.long, (self.max_visible_player_num,)),
            'enemy_in_white_safetyarea': (torch.long, (self.max_visible_player_num,)),
            'whether_enemy_run_in_blue_circle_time': (torch.float, (self.max_visible_player_num,)),
            'whether_enemy_run_in_blue_circle': (torch.float, (self.max_visible_player_num,)),
            'whether_enemy_run_in_white_circle': (torch.float, (self.max_visible_player_num,)),
 
            "hold_gun" : (torch.long, (self.max_visible_player_num,)),
            'enemy_buff_1': (torch.long, (self.max_visible_player_num,)),
            'enemy_buff_2': (torch.long, (self.max_visible_player_num,)),
            'enemy_buff_3': (torch.long, (self.max_visible_player_num,)),           
            'enemy_item_num': (torch.long, ()),
            'act_attention_shift':  (torch.float, ()),
 
        }



        self.SUPPLY_ITEM_INFO = {
            'distance': (torch.float, (self.Max_Init_Supply_Num,)),
            'quantity': (torch.float, (self.Max_Init_Supply_Num,)),
            'attribute': (torch.long, (self.Max_Init_Supply_Num,)),
            'pos_x': (torch.float, (self.Max_Init_Supply_Num,)),
            'pos_y': (torch.float, (self.Max_Init_Supply_Num,)),
            'pos_z': (torch.float, (self.Max_Init_Supply_Num,)),
            'relative_pos_x': (torch.float, (self.Max_Init_Supply_Num,)),
            'relative_pos_y': (torch.float, (self.Max_Init_Supply_Num,)),
            'relative_pos_z': (torch.float, (self.Max_Init_Supply_Num,)),
            # 'pitch': (torch.float, (self.max_supply_item_num,)),
            # 'yaw': (torch.float, (self.max_supply_item_num,)),
            'air_drop':(torch.long, (self.Max_Init_Supply_Num,)),
            'main_type': (torch.long, (self.Max_Init_Supply_Num,)),
            'subtype': (torch.long, (self.Max_Init_Supply_Num,)),
            'sub_id': (torch.long, (self.Max_Init_Supply_Num,)),
            'size': (torch.float, (self.Max_Init_Supply_Num,)),
            'supply_item_num': (torch.long, ()),
        }


        self.HEATMAP_INFO = (torch.float, (self.kernel_size, *self.heat_map_model_size))


        self.BACKPACK_INFO = {
            'main_type': (torch.long, (self.max_backpack_item_num,)),
            'subtype': (torch.long, (self.max_backpack_item_num,)),        
            'sub_id': (torch.long, (self.max_backpack_item_num,)),          
            'count': (torch.long, (self.max_backpack_item_num,)),
            'size': (torch.float, (self.max_backpack_item_num,)),
            'used_in_slot' :(torch.long, (self.max_backpack_item_num,)),
            'slot_0': (torch.long, (self.max_backpack_item_num,)),
            'slot_1': (torch.long, (self.max_backpack_item_num,)),
            'slot_2': (torch.long, (self.max_backpack_item_num,)),
            'slot_3': (torch.long, (self.max_backpack_item_num,)),
            'slot_4': (torch.long, (self.max_backpack_item_num,)),
            'backpack_item_num': (torch.long, ()),            
            # 'backpack_volume_used': torch.tensor(volume_used, dtype=torch.long),# 背包已使用量 可除200            
            

        }
        
        self.MONSTER_INFO = {

            
            'mon2own_distance': (torch.float, (self.max_monster_num,)),
            'mon_type': (torch.long, (self.max_monster_num,)),
            'mon_max_hp': (torch.float, (self.max_monster_num,)),
            'mon_cur_hp': (torch.float, (self.max_monster_num,)),
            'mon_cur_hp_percent': (torch.float, (self.max_monster_num,)),
            'mon_pos_x': (torch.float, (self.max_monster_num,)),
            'mon_pos_y': (torch.float, (self.max_monster_num,)),
            'mon_pos_z': (torch.float, (self.max_monster_num,)),
            'mon_relative_me_pos_x': (torch.float, (self.max_monster_num,)),
            'mon_relative_me_pos_y': (torch.float, (self.max_monster_num,)),
            'mon_relative_me_pos_z': (torch.float, (self.max_monster_num,)),
            "mon_rotation_x":(torch.float, (self.max_monster_num,)),
            "mon_rotation_x_sin":(torch.float, (self.max_monster_num,)),
            "mon_rotation_x_cos":(torch.float, (self.max_monster_num,)),
            "mon_rotation_y":(torch.float, (self.max_monster_num,)),
            "mon_rotation_y_sin":(torch.float, (self.max_monster_num,)),
            "mon_rotation_y_cos":(torch.float, (self.max_monster_num,)),
            "mon_rotation_z":(torch.float, (self.max_monster_num,)),
            "mon_rotation_z_sin":(torch.float, (self.max_monster_num,)),
            "mon_rotation_z_cos":(torch.float, (self.max_monster_num,)),
            "mon_size_x":(torch.float, (self.max_monster_num,)),
            "mon_size_y":(torch.float, (self.max_monster_num,)),
            "mon_size_z":(torch.float, (self.max_monster_num,)),
            "mon_target_player":(torch.long, (self.max_monster_num,)),
            "monster_num":(torch.long, ()), 
            
            
        }
        
        self.WEAPON_INFO = {
            'is_active': (torch.long, (self.max_player_weapon_num,)),
            'maintype': (torch.long, (self.max_player_weapon_num,)),
            'subtype': (torch.long, (self.max_player_weapon_num,)),
            'sub_id': (torch.long, (self.max_player_weapon_num,)),
            'bullet_current': (torch.float, (self.max_player_weapon_num,)),
            'bullet_rest': (torch.float, (self.max_player_weapon_num,)),
            'capacity': (torch.float, (self.max_player_weapon_num,)),
            'bullet_percent': (torch.float, (self.max_player_weapon_num,)),
            # 'bullet2backpack_percent': all_player_weapon[:, 8].long(), # 子弹占弹备弹百分比
            'remain_reloading': (torch.float, (self.max_player_weapon_num,)),
            # 'can_use_attachments': all_player_weapon[:,8:25].long(), # 该枪可使用哪些配件 可移除
            # 'used_attachments': all_player_weapon[:,25:41].long(), # 该枪已装配上哪些配件 可移除

            'Muzzle_main_type': (torch.long, (self.max_player_weapon_num,)),
            'Muzzle_subtype':(torch.long, (self.max_player_weapon_num,)),
            'Muzzle_sub_id': (torch.long, (self.max_player_weapon_num,)),
            'grip_main_type': (torch.long, (self.max_player_weapon_num,)),
            'grip_subtypee': (torch.long, (self.max_player_weapon_num,)),
            'grip_sub_id': (torch.long, (self.max_player_weapon_num,)),
            'butt_main_type': (torch.long, (self.max_player_weapon_num,)),
            'butt_subtype': (torch.long, (self.max_player_weapon_num,)),
            'butt_sub_id': (torch.long, (self.max_player_weapon_num,)),
            'clip_main_type': (torch.long, (self.max_player_weapon_num,)),
            'clip_subtype': (torch.long, (self.max_player_weapon_num,)),
            'clip_sub_id': (torch.long, (self.max_player_weapon_num,)),
            'sight_main_type':(torch.long, (self.max_player_weapon_num,)),
            'sight_subtype': (torch.long, (self.max_player_weapon_num,)),
            'sight_sub_id': (torch.long, (self.max_player_weapon_num,)),
            'bullet_main_type': (torch.long, (self.max_player_weapon_num,)),
            'bullet_subtype':(torch.long, (self.max_player_weapon_num,)),
            'bullet_sub_id': (torch.long, (self.max_player_weapon_num,)),
            'player_weapon_num': (torch.long, ()),   


        }

        self.DOOR_INFO = {
            'distance': (torch.float, (self.max_door_num,)),
            'x': (torch.float, (self.max_door_num,)),
            'y': (torch.float, (self.max_door_num,)),
            'z': (torch.float, (self.max_door_num,)),
            're_x': (torch.float, (self.max_door_num,)),
            're_y': (torch.float, (self.max_door_num,)),
            're_z': (torch.float, (self.max_door_num,)),
            'door_state': (torch.long, (self.max_door_num,)),
            'pitch': (torch.float, (self.max_door_num,)),
            'yaw': (torch.float, (self.max_door_num,)),
            'door_type': (torch.long, (self.max_door_num,)),
            'door_num': (torch.long, ()),
        }


        # self.SPATIAL_INFO = (torch.float, (1, *self.spatial_size))

  
        self.ONLY_V_INFO = {
            'distance': (torch.float, (self.Max_Init_Enemy_Num,)),
            'team_id': (torch.long, (self.Max_Init_Enemy_Num,)),
            'pos_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'pos_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'pos_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_sin_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_sin_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_sin_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_cos_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_cos_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'rotation_cos_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'size_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'size_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'size_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'speed_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'speed_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'speed_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'scalar_speed': (torch.float, (self.Max_Init_Enemy_Num,)),
            'hp': (torch.float, (self.Max_Init_Enemy_Num,)),
            'neardeath_breath': (torch.float, (self.Max_Init_Enemy_Num,)),
            'oxygen': (torch.float, (self.Max_Init_Enemy_Num,)),
            'peek': (torch.long, (self.Max_Init_Enemy_Num,)),
            'alive': (torch.long, (self.Max_Init_Enemy_Num,)),
            'bodystate': (torch.long, (self.Max_Init_Enemy_Num,)),
            'relative_pos_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'relative_pos_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'relative_pos_z': (torch.float, (self.Max_Init_Enemy_Num,)),
            'character': (torch.long, (self.Max_Init_Enemy_Num,)),          
            'enemy_see_me': (torch.long, (self.Max_Init_Enemy_Num,)),           
            'enemy_relative_blue_safetyarea_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_relative_blue_safetyarea_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_relative_white_safetyarea_x': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_relative_white_safetyarea_y': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_distance_blue_safetyarea': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_distance_blue_safetyarea_relative': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_distance_white_safetyarea': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_distance_white_safetyarea_relative': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_in_blue_safetyarea': (torch.long, (self.Max_Init_Enemy_Num,)),
            'enemy_in_white_safetyarea': (torch.long, (self.Max_Init_Enemy_Num,)),
            'whether_enemy_run_in_blue_circle_time': (torch.float, (self.Max_Init_Enemy_Num,)),
            'whether_enemy_run_in_blue_circle': (torch.float, (self.Max_Init_Enemy_Num,)),
            'whether_enemy_run_in_white_circle': (torch.float, (self.Max_Init_Enemy_Num,)),
            "hold_gun" : (torch.long, (self.Max_Init_Enemy_Num,)),
            'since_last_see_time': (torch.float, (self.Max_Init_Enemy_Num,)),
            'enemy_buff_1': (torch.long, (self.Max_Init_Enemy_Num,)),
            'enemy_buff_2': (torch.long, (self.Max_Init_Enemy_Num,)),
            'enemy_buff_3': (torch.long, (self.Max_Init_Enemy_Num,)),          
            'enemy_item_num': (torch.long, ()),
            
        }
    
        
        self.ADVANTAGE = {k:(torch.float, ()) for k in self.cfg.agent.enable_baselines}
        self.RETURN = {k:(torch.float, ()) for k in self.cfg.agent.enable_baselines}
        
        self.REWARD_INFO = {
            "search":(torch.float, ()),
            "time":(torch.float, ()),
            "bullet":(torch.float, ()),
            "hp":(torch.float, ()),
            "be_knocked_down":(torch.float, ()),
            "dead":(torch.float, ()),
            "damage_enemy":(torch.float, ()),
            "knock_down_enemy":(torch.float, ()),
            "kill_enemy":(torch.float, ()),
            "approach_knockdown_teammate":(torch.float, ()),
            "help_up_teammate":(torch.float, ()),
            "not_save_teammate":(torch.float, ()),
            "goto_circle":(torch.float, ()),
            "rank":(torch.float, ()),
            "sum":(torch.float, ()),
            "not_stand":(torch.float, ()),
            "loc_reward":(torch.float, ()),
            "supply_reward":(torch.float, ()),
            "close2supply_reward":(torch.float, ()),
            "reward_see_enemy": (torch.float, ()),
            "out_of_circle":(torch.float, ()),
            "be_seen_reward": (torch.float, ()),
            "reward_damage_teammate": (torch.float, ()),
            "reward_teamate_up": (torch.float, ()),
            "reward_teamate_realup": (torch.float, ()),
            "reward_approach_teammate": (torch.float, ()),
            "reward_abort_help_up":(torch.float, ()),
            "stuck":(torch.float, ()),
        }



        
    def get_rl_step_data(self, last=False):
        data = {}
        event_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.EVENT_INFO.items()}
        scalar_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.SCALAR_INFO.items()}
        backpack_item_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.BACKPACK_INFO.items()}
        player_weapon_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.WEAPON_INFO.items()}
        supply_item_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.SUPPLY_ITEM_INFO.items()}
        door_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.DOOR_INFO.items()}
        enemy_item_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.ENEMY_INFO.items()}
        visible_enemy_item_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.VISIBLE_ENEMY_INFO.items()}
        teammate_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.TEAMMATE_INFO.items()}
        # spatial_info = torch.zeros(size=self.SPATIAL_INFO[1], dtype=self.SPATIAL_INFO[0])
        heatmap_info = torch.zeros(size=self.HEATMAP_INFO[1], dtype=self.HEATMAP_INFO[0])
        # position_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.POSITION_INFO.items()}
        mask_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.MASK_INFO.items()}
        only_v_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.ONLY_V_INFO.items()}
        history_positions_info = torch.zeros(size=self.HISTORY_POSITIONS_INFO[1], dtype=self.HISTORY_POSITIONS_INFO[0])
        monster_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.MONSTER_INFO.items()}
        rotation_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.ROTATION_INFO.items()}
        # heatmap_info = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.HEATMAP_INFO.items()}
        data['obs'] = {'scalar_info': scalar_info,
                       'backpack_item_info': backpack_item_info,
                       'player_weapon_info': player_weapon_info,
                       'supply_item_info': supply_item_info,
                       'door_info': door_info,
                       'teammate_info': teammate_info,
                       "enemy_item_info":enemy_item_info,
                       "visible_enemy_item_info":visible_enemy_item_info,
                       "monster_info":monster_info,
                       "event_info":event_info,
                    #  'spatial_info': spatial_info,
                     'mask_info': mask_info, 
                     'only_v_info': only_v_info, 
                     "heatmap_info": heatmap_info,
                     "history_positions_info":history_positions_info,
                     "rotation_info":rotation_info,
                    #  "position_info":position_info,
                     }
        if not last:
            data['action'] = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.ACTION_INFO.items()}
            data['action_logp'] = {k: torch.ones(size=v[1], dtype=v[0]) for k, v in self.ACTION_LOGP.items()}
            data['reward'] = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.REWARD_INFO.items()}
            data['done'] = torch.zeros(size=(), dtype=torch.bool)
            data['model_last_iter'] = torch.zeros(size=(), dtype=torch.float)
            data['advantage'] = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.ADVANTAGE.items()}
            data['return'] = {k: torch.zeros(size=v[1], dtype=v[0]) for k, v in self.RETURN.items()}
        return data

    def get_rl_traj_data(self, unroll_len):
        traj_data_list = []
        for _ in range(unroll_len):
            traj_data_list.append(self.get_rl_step_data())
        traj_data_list.append(self.get_rl_step_data(last=True))
        traj_data = default_collate_with_dim(traj_data_list)
        return traj_data

    def get_rl_batch_data(self, unroll_len, batch_size):
        batch_data_list = []
        for _ in range(batch_size):
            batch_data_list.append(self.get_rl_traj_data(unroll_len))
        batch_data = default_collate_with_dim(batch_data_list, dim=1)
        return batch_data

    def speed_vec_to_scalar(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    def computer_distance(self, x, y, z, target_x, target_y, target_z):
        return math.sqrt((target_x - x)**2 + (target_y - y)**2 + (target_z-z)**2)

    def vec_of_2position(self, start_x, start_y, start_z, target_x, target_y, target_z):
        dx = target_x - start_x
        dy = target_y - start_y
        dz = target_z - start_z
        dir_vec = np.asarray([dx, dy, dz])
        return (dir_vec / (np.linalg.norm(dir_vec) + 1e-9)).tolist()

    def player_in_safetyarea(self, player_x, player_y, circle_x, circle_y, radius):
        return self.computer_distance(player_x, player_y,0, circle_x, circle_y,0) < radius

    def find_nearest_supplys(self,pos):
        dis_matrix = (torch.as_tensor(pos) - self.supplys_pos_matrix).norm(p=2,dim=-1)
        index = dis_matrix.le(self.supply_visble_distance)
        return self.supplys_categorys[index]

    
    
    
    def transform_scalar_info(self,state_info, info, state, progress_bar_info, last_action):
           
        # TY
        own_player_state = state_info[self.id]
        own_character = all_characters[own_player_state.state.actor_id].id - 1100
        own_player_speed_scalar = self.speed_vec_to_scalar(own_player_state.state.speed.x,
                                                own_player_state.state.speed.y,
                                                own_player_state.state.speed.z)
        own_player_speed_scalar = min(own_player_speed_scalar, 1000)
        own_skill_buff = [0,0,0]
        # own_player_buff = [0] * 11
        for skill_idx,i in enumerate(own_player_state.state.buff):
            if skill_idx >= 3:
                break 
            own_skill_buff[skill_idx] = i+1
        

        if self.teammate_dict is None:
            self.teammate_dict = defaultdict(list)
            for i in info['player_state'].keys():
                for j in info['player_state'].keys():
                    if i != j and info['player_state'][i].state.team_id == info['player_state'][j].state.team_id:
                        self.teammate_dict[i].append(j)

        safety_area = state.safety_area
        own_safety_area_state = safety_area.state - 1
        own_safety_area_pos_x = safety_area.center.x 
        own_safety_area_pos_y = safety_area.center.y 
        own_safety_area_radius = safety_area.radius 
        own_safety_area_next_pos_x = safety_area.next_center.x 
        own_safety_area_next_pos_y = safety_area.next_center.y 
        own_safety_area_next_radius = safety_area.next_radius 
        own_safety_area_rest_time = safety_area.total_time - safety_area.time  # rest_time  TODO
        # 安全区伤害
        safezone_pain = safety_area.pain
        safezone_appear_time = safety_area.safezone_appear_time
        safezone_delay_time = safety_area.delay_time
        
        own_player_in_blue_safetyarea = self.player_in_safetyarea(own_player_state.state.position.x,
                                                              own_player_state.state.position.y,
                                                              safety_area.center.x,
                                                              safety_area.center.y,
                                                              safety_area.radius)
        own_player_in_white_safetyarea = self.player_in_safetyarea(own_player_state.state.position.x,
                                                               own_player_state.state.position.y,
                                                               safety_area.next_center.x,
                                                               safety_area.next_center.y,
                                                               safety_area.next_radius, )
        own_player_vec_blue_safetyarea_x, own_player_vec_blue_safetyarea_y,_  = self.vec_of_2position(own_player_state.state.position.x / self.mapsize[0],
                                                             own_player_state.state.position.y / self.mapsize[1],
                                                             0,
                                                             safety_area.center.x / self.mapsize[0],
                                                             safety_area.center.y / self.mapsize[1], 0,)

        own_player_vec_white_safetyarea_x, own_player_vec_white_safetyarea_y,_ = self.vec_of_2position(own_player_state.state.position.x / self.mapsize[0],
                                                              own_player_state.state.position.y / self.mapsize[1],0,
                                                              safety_area.next_center.x / self.mapsize[0],
                                                              safety_area.next_center.y / self.mapsize[1], 0,)
        #TODO
        own_dis_blue_safetyarea = self.computer_distance(own_player_state.state.position.x,
                                                              own_player_state.state.position.y,0,
                                                              safety_area.center.x, safety_area.center.y, 0)

        own_dis_white_safetyarea = self.computer_distance(own_player_state.state.position.x,
                                                               own_player_state.state.position.y,0,
                                                               safety_area.next_center.x,
                                                               safety_area.next_center.y,0)
        #TODO clip
        own_whether_run_in_blue_circle_time = ((own_dis_blue_safetyarea - safety_area.radius) / (
                                                           self.expect_speed ) - (safety_area.total_time - safety_area.time)) / self.MAX_GAME_TIME

        own_whether_run_in_blue_circle = (own_dis_blue_safetyarea - safety_area.radius) / (self.expect_speed)

        own_whether_run_in_white_circle = (own_dis_white_safetyarea - safety_area.next_radius) / (self.expect_speed)

        if self.last_hp is None:
            hp_delta = 0
            neardeath_hp_delta = 0
        else:
            hp_delta = own_player_state.state.hp - self.last_hp
            neardeath_hp_delta = own_player_state.state.neardeath_breath - self.last_neardeath_hp

        
        
        # print(f'self_id:{self.id}, rotation x:{own_player_state.state.rotation.x }, rotation y:{own_player_state.state.rotation.y}, rotation z:{own_player_state.state.rotation.z}')
        # print(f'self_id:{self.id}, position x:{own_player_state.state.position.x }, position y:{own_player_state.state.position.y}, position z:{own_player_state.state.position.z}')
        # if hp_delta != 0 or neardeath_hp_delta !=0:
        #     print(f'current hp:{own_player_state.state.hp}, last hp:{self.last_hp}, current neardeath hp:{own_player_state.state.neardeath_breath}, last neardeath hp:{self.last_neardeath_hp}')
        
        self.last_hp = own_player_state.state.hp
        self.last_neardeath_hp = own_player_state.state.neardeath_breath
        # 处理pitch
        pitch_angle = own_player_state.camera.rotation.y 
        pitch_angle = pitch_angle if pitch_angle <= 90 else pitch_angle - 360



    
 
        last_action = last_action["single_head"].item() if last_action is not None else 0
        scalar_info = {
            "is_treat":torch.tensor(progress_bar_info[0], dtype=torch.long),
            "treat_remain_time":torch.tensor(progress_bar_info[1]/self.reloading_time_norm, dtype=torch.float),
            "is_rescue":torch.tensor(progress_bar_info[2], dtype=torch.long),
            "rescue_remain_time":torch.tensor(progress_bar_info[3]/self.reloading_time_norm, dtype=torch.float),
            "is_rescued":torch.tensor(progress_bar_info[4], dtype=torch.long),
            "rescued_remain_time":torch.tensor(progress_bar_info[5]/self.reloading_time_norm, dtype=torch.float),
            "is_reloading":torch.tensor(progress_bar_info[6], dtype=torch.long),
            "reloading_remain_time":torch.tensor(progress_bar_info[7]/self.reloading_time_norm, dtype=torch.float),
            "own_safety_area_state": torch.tensor(own_safety_area_state, dtype=torch.long),
            "own_safety_area_pos_x":torch.tensor(own_safety_area_pos_x / self.mapsize[0], dtype=torch.float),
            "own_safety_area_pos_y":torch.tensor(own_safety_area_pos_y / self.mapsize[1], dtype=torch.float),
            "own_safety_area_radius":torch.tensor(own_safety_area_radius / self.mapsize[0], dtype=torch.float),
            "own_safety_area_next_pos_x":torch.tensor(own_safety_area_next_pos_x / self.mapsize[0], dtype=torch.float),
            "own_safety_area_next_pos_y":torch.tensor(own_safety_area_next_pos_y / self.mapsize[1], dtype=torch.float),
            "own_safety_area_next_radius":torch.tensor(own_safety_area_next_radius / self.mapsize[0], dtype=torch.float),
            "safety_area_time": torch.tensor(safety_area.time / self.MAX_GAME_TIME, dtype=torch.float),
            "safety_area_total_time": torch.tensor(safety_area.total_time / self.MAX_GAME_TIME, dtype=torch.float),
            "own_safety_area_rest_time":torch.tensor(own_safety_area_rest_time / self.MAX_GAME_TIME, dtype=torch.float),
            "own_player_in_blue_safetyarea":torch.tensor(own_player_in_blue_safetyarea, dtype=torch.long),
            "own_player_in_white_safetyarea":torch.tensor(own_player_in_white_safetyarea, dtype=torch.long),
            "own_player_vec_blue_safetyarea_x":torch.tensor(own_player_vec_blue_safetyarea_x, dtype=torch.float),
            "own_player_vec_blue_safetyarea_y":torch.tensor(own_player_vec_blue_safetyarea_y, dtype=torch.float),
            "own_player_vec_white_safetyarea_x":torch.tensor(own_player_vec_white_safetyarea_x, dtype=torch.float),
            "own_player_vec_white_safetyarea_y":torch.tensor(own_player_vec_white_safetyarea_y, dtype=torch.float),
            "own_dis_blue_safetyarea":torch.tensor(own_dis_blue_safetyarea / self.mapsize[0], dtype=torch.float),
            "own_dis_white_safetyarea":torch.tensor(own_dis_white_safetyarea / self.mapsize[0], dtype=torch.float),
            "own_dis_blue_safetyarea_radius":torch.tensor(own_dis_blue_safetyarea / (own_safety_area_radius + 1), dtype=torch.float),
            "own_dis_white_safetyarea_radius":torch.tensor(own_dis_white_safetyarea / (own_safety_area_next_radius+ 1), dtype=torch.float),
            "own_whether_run_in_blue_circle_time":torch.tensor(own_whether_run_in_blue_circle_time, dtype=torch.float).clamp(min=-5,max=5),
            "own_whether_run_in_blue_circle":torch.tensor(own_whether_run_in_blue_circle/ self.MAX_GAME_TIME, dtype=torch.float).clamp(min=-5,max=5),
            "own_whether_run_in_white_circle":torch.tensor(own_whether_run_in_white_circle/ self.MAX_GAME_TIME, dtype=torch.float).clamp(min=-5,max=5),
            "safezone_pain":torch.tensor(safezone_pain / self.pain_norm, dtype=torch.float),
            "safezone_appear_time":torch.tensor(safezone_appear_time / self.MAX_GAME_TIME, dtype=torch.float),
            "safezone_delay_time":torch.tensor(safezone_delay_time / self.MAX_GAME_TIME, dtype=torch.float),
            
            'character_id': torch.tensor(own_character, dtype=torch.long), # binary 5
            'team_id' : torch.tensor(own_player_state.state.team_id, dtype=torch.long), #binaty 5
            'position_x': torch.tensor(own_player_state.state.position.x / self.mapsize[0], dtype=torch.float),
            'position_y': torch.tensor(own_player_state.state.position.y / self.mapsize[1], dtype=torch.float),
            'position_z': torch.tensor(own_player_state.state.position.z / self.mapsize[2], dtype=torch.float),

            #TODO
            'rotation_x': torch.tensor(own_player_state.state.rotation.x / self.rotation_norm, dtype=torch.float),
            'rotation_y': torch.tensor(own_player_state.state.rotation.y / self.rotation_norm, dtype=torch.float),
            'rotation_z': torch.tensor(own_player_state.state.rotation.z / self.rotation_norm, dtype=torch.float),
            'sin_rotation_x':torch.tensor(math.sin(own_player_state.state.rotation.x*math.pi/180), dtype=torch.float),
            'cos_rotation_x':torch.tensor(math.cos(own_player_state.state.rotation.x*math.pi/180), dtype=torch.float),
            'sin_rotation_y':torch.tensor(math.sin(own_player_state.state.rotation.y*math.pi/180), dtype=torch.float),
            'cos_rotation_y':torch.tensor(math.cos(own_player_state.state.rotation.y*math.pi/180), dtype=torch.float),
            'sin_rotation_z':torch.tensor(math.sin(own_player_state.state.rotation.z*math.pi/180), dtype=torch.float),
            'cos_rotation_z':torch.tensor(math.cos(own_player_state.state.rotation.z*math.pi/180), dtype=torch.float),

            'size_x': torch.tensor(own_player_state.state.size.x / self.size_xy_norm, dtype=torch.float),
            'size_y': torch.tensor(own_player_state.state.size.y / self.size_xy_norm, dtype=torch.float),
            'size_z': torch.tensor(own_player_state.state.size.z / self.size_z_norm, dtype=torch.float),

            'speed_x': torch.tensor(own_player_state.state.speed.x / self.speed_norm, dtype=torch.float),
            'speed_y': torch.tensor(own_player_state.state.speed.y / self.speed_norm, dtype=torch.float),
            'speed_z': torch.tensor(own_player_state.state.speed.z / self.speed_norm, dtype=torch.float),
            'speed_scalar': torch.tensor(own_player_speed_scalar / 750, dtype=torch.float),

            'hp': torch.tensor(own_player_state.state.hp / self.hp_norm, dtype=torch.float),
            # "hp_delta": torch.tensor(hp_delta / self.hp_norm, dtype=torch.float),
            'neardeath_breath': torch.tensor(own_player_state.state.neardeath_breath / self.hp_norm, dtype=torch.float),
            'oxygen': torch.tensor(own_player_state.state.oxygen/self.oxygen_norm, dtype=torch.float),
            # 'buff': torch.tensor(own_player_buff , dtype=torch.long), #11 tensor
            'peek_type': torch.tensor(own_player_state.state.peek_type, dtype=torch.long), # one hot 3
            'alive_state': torch.tensor(own_player_state.state.alive_state, dtype=torch.long),# one hot 3
            'body_state': torch.tensor(own_player_state.state.body_state, dtype=torch.long),# one hot 8

            'is_switching': torch.tensor(own_player_state.state.is_switching, dtype=torch.long),
            'is_pose_changing': torch.tensor(own_player_state.state.is_pose_changing, dtype=torch.long),
            'is_running': torch.tensor(own_player_state.state.is_running, dtype=torch.long),
            'is_aiming': torch.tensor(own_player_state.state.is_aiming, dtype=torch.long),
            'is_firing': torch.tensor(own_player_state.state.is_firing, dtype=torch.long),
            'is_holding': torch.tensor(own_player_state.state.is_holding, dtype=torch.long),
            'is_falling': torch.tensor(own_player_state.state.is_falling, dtype=torch.long),
            'is_picking': torch.tensor(own_player_state.state.is_picking, dtype=torch.long),

            'camera_x': torch.tensor((own_player_state.camera.position.x - own_player_state.state.position.x)/ self.camera_position_norm,   ##TODO
                                     dtype=torch.float).clamp(min=-1,max=1),
            'camera_y': torch.tensor((own_player_state.camera.position.y - own_player_state.state.position.y)/ self.camera_position_norm,
                                     dtype=torch.float).clamp(min=-1,max=1),
            'camera_z': torch.tensor((own_player_state.camera.position.z - own_player_state.state.position.z)/ self.camera_position_norm,
                                     dtype=torch.float).clamp(min=-1,max=1),

            'camera_rotation_x': torch.tensor(own_player_state.camera.rotation.x / self.rotation_norm, dtype=torch.float),
            'camera_rotation_y': torch.tensor(pitch_angle / 90, dtype=torch.float),
            'camera_rotation_z': torch.tensor(own_player_state.camera.rotation.z / self.rotation_norm, dtype=torch.float),
            'sin_camera_rotation_x':torch.tensor(math.sin(own_player_state.camera.rotation.x*math.pi/180), dtype=torch.float),
            'cos_camera_rotation_x':torch.tensor(math.cos(own_player_state.camera.rotation.x*math.pi/180), dtype=torch.float),
            'sin_camera_rotation_y':torch.tensor(math.sin(own_player_state.camera.rotation.y*math.pi/180), dtype=torch.float),
            'cos_camera_rotation_y':torch.tensor(math.cos(own_player_state.camera.rotation.y*math.pi/180), dtype=torch.float),
            'sin_camera_rotation_z':torch.tensor(math.sin(own_player_state.camera.rotation.z*math.pi/180), dtype=torch.float),
            'cos_camera_rotation_z':torch.tensor(math.cos(own_player_state.camera.rotation.z*math.pi/180), dtype=torch.float),
            'skill_buff_1': torch.tensor(own_skill_buff[0], dtype=torch.long),
            'skill_buff_2': torch.tensor(own_skill_buff[1], dtype=torch.long),
            'skill_buff_3': torch.tensor(own_skill_buff[2], dtype=torch.long),
            'last_action':torch.tensor(last_action, dtype=torch.long),

            'dir_distance_1':torch.tensor(min(own_player_state.dirs_ray_distance[0] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_2':torch.tensor(min(own_player_state.dirs_ray_distance[1] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_3':torch.tensor(min(own_player_state.dirs_ray_distance[2] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_4':torch.tensor(min(own_player_state.dirs_ray_distance[3] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_5':torch.tensor(min(own_player_state.dirs_ray_distance[4] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_6':torch.tensor(min(own_player_state.dirs_ray_distance[5] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_7':torch.tensor(min(own_player_state.dirs_ray_distance[6] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_8':torch.tensor(min(own_player_state.dirs_ray_distance[7] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_9':torch.tensor(min(own_player_state.dirs_ray_distance[8] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_10':torch.tensor(min(own_player_state.dirs_ray_distance[9] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_11':torch.tensor(min(own_player_state.dirs_ray_distance[10] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_12':torch.tensor(min(own_player_state.dirs_ray_distance[11] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_13':torch.tensor(min(own_player_state.dirs_ray_distance[12] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_14':torch.tensor(min(own_player_state.dirs_ray_distance[13] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_15':torch.tensor(min(own_player_state.dirs_ray_distance[14] / self.ray_dirs_norm,1.5), dtype=torch.float),
            'dir_distance_16':torch.tensor(min(own_player_state.dirs_ray_distance[15] / self.ray_dirs_norm,1.5), dtype=torch.float),
            # 'teammate_visible': torch.tensor(teammate_visible , dtype=torch.long) # team players in one team
        }
        swimming_mask = own_player_state.state.body_state == 6
        self.pitch_angle = pitch_angle
        self.yaw_angle = own_player_state.camera.rotation.z
        return scalar_info,swimming_mask
    
    def transform_teammate_info(self, info, own_player_state,state_info, safety_area):
                # ============
        # teammate info
        # ============
        teammate_info = [[0 for _ in range(68)] for _ in range(self.teammate_num)]
        team_idx = sorted(self.teammate_dict[self.id])
        
        teammate_alive_state = 0
        self.need_rescued_teammate_id = None
        wait2saved_dist = np.inf
        # heatmap teammate info 
        top2bottom_teammate_position = [[] for _ in range(len(self.heat_map_deltas["top2bottom_10"]))]
        pitch_teammate_position = [[] for _ in range(len(self.heat_map_deltas["pitch_delta_10"]))]
        yaw_teammate_position = [[] for _ in range(len(self.heat_map_deltas["yaw_delta_10"]))]
        self.alive_teammate_nums = 0
        for k,v in info['alive_players'].items():
            if k == self.id:
                self.alive_teammate_nums += 1
                continue
            if v and info['player_state'][k].state.team_id == own_player_state.state.team_id:
                
                teammate_k_obs = info['player_state'][k]
                if teammate_k_obs.state.alive_state == 1:
                    self.alive_teammate_nums += 1
                    teammate_alive_state = 1
                    be_rescued_dist = self.computer_distance(own_player_state.state.position.x ,
                                                                own_player_state.state.position.y ,
                                                                own_player_state.state.position.z ,
                                                                teammate_k_obs.state.position.x ,
                                                                teammate_k_obs.state.position.y ,
                                                                teammate_k_obs.state.position.z , )
                    if wait2saved_dist > be_rescued_dist:
                        self.need_rescued_teammate_id = k
                        wait2saved_dist = be_rescued_dist
                        
                for delta,pool_map_size in self.all_pool_map_size.items():
                    if abs(own_player_state.state.position.x-teammate_k_obs.state.position.x) <= pool_map_size[0]*self.heat_map_scale \
                        and abs(own_player_state.state.position.y-teammate_k_obs.state.position.y) <= pool_map_size[1]*self.heat_map_scale:
                        pos = [int(teammate_k_obs.state.position.x//self.heat_map_scale),
                                int(teammate_k_obs.state.position.y//self.heat_map_scale)]
                        if delta in self.heat_map_deltas["top2bottom_10"]:
                            top2bottom_teammate_position[self.heat_map_deltas["top2bottom_10"].index(delta)].append(pos)
                        if delta in self.heat_map_deltas["pitch_delta_10"]:
                            pitch_teammate_position[self.heat_map_deltas["pitch_delta_10"].index(delta)].append(pos)
                        if delta in self.heat_map_deltas["yaw_delta_10"]:
                            yaw_teammate_position[self.heat_map_deltas["yaw_delta_10"].index(delta)].append(pos)
                
                teammate_character = all_characters[teammate_k_obs.state.actor_id]

                teammate_pos_x = teammate_k_obs.state.position.x / self.mapsize[0]
                teammate_pos_y = teammate_k_obs.state.position.y / self.mapsize[1]
                teammate_pos_z = teammate_k_obs.state.position.z / self.mapsize[2]
                teammate_rotation_x = teammate_k_obs.state.rotation.x / self.rotation_norm
                teammate_rotation_y = teammate_k_obs.state.rotation.y / self.rotation_norm
                teammate_rotation_z = teammate_k_obs.state.rotation.z / self.rotation_norm
                sin_rotation_x = math.sin(teammate_k_obs.state.rotation.x*math.pi/180)
                cos_rotation_x = math.cos(teammate_k_obs.state.rotation.x*math.pi/180)
                sin_rotation_y = math.sin(teammate_k_obs.state.rotation.y*math.pi/180)
                cos_rotation_y = math.cos(teammate_k_obs.state.rotation.y*math.pi/180)
                sin_rotation_z = math.sin(teammate_k_obs.state.rotation.z*math.pi/180)
                cos_rotation_z = math.cos(teammate_k_obs.state.rotation.z*math.pi/180)
                teammate_size_x = teammate_k_obs.state.size.x / self.size_xy_norm
                teammate_size_y = teammate_k_obs.state.size.y / self.size_xy_norm
                teammate_size_z = teammate_k_obs.state.size.z / self.size_z_norm
                teammate_speed_x = teammate_k_obs.state.speed.x / self.speed_norm
                teammate_speed_y = teammate_k_obs.state.speed.y / self.speed_norm
                teammate_speed_z = teammate_k_obs.state.speed.z / self.speed_norm
                teammate_scalar_speed = self.speed_vec_to_scalar(teammate_k_obs.state.speed.x,
                                                                 teammate_k_obs.state.speed.y,
                                                                 teammate_k_obs.state.speed.z)
                teammate_scalar_speed = min(teammate_scalar_speed, 1000)
                teammate_hp = teammate_k_obs.state.hp / self.hp_norm
                teammate_neardeath_breath = teammate_k_obs.state.neardeath_breath / self.hp_norm
                teammate_oxygen = teammate_k_obs.state.oxygen / self.oxygen_norm
                teammate_peek = teammate_k_obs.state.peek_type 
                teammate_alive = teammate_k_obs.state.alive_state 
                teammate_bodystate = teammate_k_obs.state.body_state 
                teammate_camera_position_x = (teammate_k_obs.camera.position.x - own_player_state.state.position.x)
                teammate_camera_position_y = (teammate_k_obs.camera.position.y - own_player_state.state.position.y)
                teammate_camera_position_z = (teammate_k_obs.camera.position.z - own_player_state.state.position.z)
                teammate_camera_rotation_x = teammate_k_obs.camera.rotation.x / self.rotation_norm
                teammate_camera_rotation_y = teammate_k_obs.camera.rotation.y / self.rotation_norm
                teammate_camera_rotation_z = teammate_k_obs.camera.rotation.z / self.rotation_norm
                sin_camera_rotation_x = math.sin(teammate_k_obs.camera.rotation.x*math.pi/180)
                cos_camera_rotation_x = math.cos(teammate_k_obs.camera.rotation.x*math.pi/180)
                sin_camera_rotation_y = math.sin(teammate_k_obs.camera.rotation.y*math.pi/180)
                cos_camera_rotation_y = math.cos(teammate_k_obs.camera.rotation.y*math.pi/180)
                sin_camera_rotation_z = math.sin(teammate_k_obs.camera.rotation.z*math.pi/180)
                cos_camera_rotation_z = math.cos(teammate_k_obs.camera.rotation.z*math.pi/180)
                teammate_player_vec_x, teammate_player_vec_y, teammate_player_vec_z = self.vec_of_2position(own_player_state.state.position.x ,
                                                               own_player_state.state.position.y ,
                                                               own_player_state.state.position.z ,
                                                               teammate_k_obs.state.position.x ,
                                                               teammate_k_obs.state.position.y ,
                                                               teammate_k_obs.state.position.z , )
                teammate_player_dis = self.computer_distance(own_player_state.state.position.x  ,
                                                                own_player_state.state.position.y ,
                                                                own_player_state.state.position.z  ,
                                                                teammate_k_obs.state.position.x ,
                                                                teammate_k_obs.state.position.y ,
                                                                teammate_k_obs.state.position.z  , )
                
                
                try:
                    teammate_can_see_me = self.id in state_info[k].visble_player_ids
                except:
                    teammate_can_see_me = False
                    print("Visble_player_ids generate error!!!!!!!!!!!!!!!!!!!!!!")

                player_in_blue_safetyarea = self.player_in_safetyarea(teammate_k_obs.state.position.x,
                                                                      teammate_k_obs.state.position.y,
                                                                      safety_area.center.x,
                                                                      safety_area.center.y,
                                                                      safety_area.radius)
                player_in_white_safetyarea = self.player_in_safetyarea(teammate_k_obs.state.position.x,
                                                                      teammate_k_obs.state.position.y,
                                                                      safety_area.next_center.x,
                                                                      safety_area.next_center.y,
                                                                      safety_area.next_radius, )
                player_vec_blue_safetyarea_x, player_vec_blue_safetyarea_y,_ = self.vec_of_2position(teammate_k_obs.state.position.x / self.mapsize[0],
                                                                      teammate_k_obs.state.position.y / self.mapsize[1],0,
                                                                      safety_area.center.x / self.mapsize[0],
                                                                      safety_area.center.y / self.mapsize[1], 0,)
                player_vec_white_safetyarea_x, player_vec_white_safetyarea_y,_ = self.vec_of_2position(teammate_k_obs.state.position.x / self.mapsize[0],
                                                                      teammate_k_obs.state.position.y / self.mapsize[1],0,
                                                                      safety_area.next_center.x / self.mapsize[0],
                                                                      safety_area.next_center.y / self.mapsize[1], 0,)

                teammate_dis_blue_safetyarea = self.computer_distance(teammate_k_obs.state.position.x, teammate_k_obs.state.position.y,0,
                                       safety_area.center.x, safety_area.center.y,0)

                teammate_dis_white_safetyarea = self.computer_distance(teammate_k_obs.state.position.x, teammate_k_obs.state.position.y,0,
                                       safety_area.next_center.x, safety_area.next_center.y,0)

                whether_teammate_run_in_circle_time = ((self.computer_distance(teammate_k_obs.state.position.x, teammate_k_obs.state.position.y,0,
                                                        safety_area.center.x, safety_area.center.y,0) - safety_area.radius) / (self.expect_speed) - (safety_area.total_time - safety_area.time)) /  self.MAX_GAME_TIME

                whether_teammate_run_in_blue_circle = (self.computer_distance(teammate_k_obs.state.position.x, teammate_k_obs.state.position.y,0,
                                       safety_area.center.x, safety_area.center.y,0) - safety_area.radius) / (self.expect_speed)

                whether_teammate_run_in_white_circle = (self.computer_distance(teammate_k_obs.state.position.x, teammate_k_obs.state.position.y,0,
                                       safety_area.next_center.x, safety_area.next_center.y,0) - safety_area.next_radius) / (self.expect_speed)

                buff_list = [0] * 3
                for buff_idx,i in enumerate(teammate_k_obs.state.buff):
                    if buff_idx >= 3:
                        break
                    buff_list[buff_idx] = i+1

            
                teammate_info[team_idx.index(k)] = [
                    teammate_character.index,
                    teammate_k_obs.state.team_id,
                    teammate_pos_x,
                    teammate_pos_y,
                    teammate_pos_z,
                    teammate_rotation_x,
                    teammate_rotation_y,
                    teammate_rotation_z,
                    teammate_size_x,
                    teammate_size_y,
                    teammate_size_z,
                    teammate_speed_x,
                    teammate_speed_y,
                    teammate_speed_z,
                    teammate_scalar_speed ,
                    teammate_hp,
                    teammate_neardeath_breath,
                    teammate_oxygen,
                    teammate_peek,
                    teammate_alive,
                    teammate_bodystate,
                    teammate_camera_position_x/(teammate_player_dis+1e-9),
                    teammate_camera_position_y/(teammate_player_dis+1e-9),
                    teammate_camera_position_z/(teammate_player_dis+1e-9),
                    teammate_camera_rotation_x,
                    teammate_camera_rotation_y,
                    teammate_camera_rotation_z,
                    sin_rotation_x,
                    cos_rotation_x,
                    sin_rotation_y,
                    cos_rotation_y,
                    sin_rotation_z ,
                    cos_rotation_z,
                    teammate_k_obs.state.is_switching,
                    teammate_k_obs.state.is_pose_changing,
                    teammate_k_obs.state.is_running,
                    teammate_k_obs.state.is_aiming,
                    teammate_k_obs.state.is_firing,
                    teammate_k_obs.state.is_holding,
                    teammate_k_obs.state.is_falling,
                    teammate_k_obs.state.is_picking,
                    teammate_player_vec_x/(teammate_player_dis+1e-9),
                    teammate_player_vec_y/(teammate_player_dis+1e-9),
                    teammate_player_vec_z/(teammate_player_dis+1e-9),
                    teammate_player_dis/self.mapsize[0],
                    # teammate_player_dis,
                    teammate_can_see_me,
                    sin_camera_rotation_x ,
                    cos_camera_rotation_x ,
                    sin_camera_rotation_y ,
                    cos_camera_rotation_y ,
                    sin_camera_rotation_z ,
                    cos_camera_rotation_z ,
                    player_in_blue_safetyarea,
                    player_in_white_safetyarea,
                    player_vec_blue_safetyarea_x,
                    player_vec_blue_safetyarea_y,
                    player_vec_white_safetyarea_x,
                    player_vec_white_safetyarea_y,
                    teammate_dis_blue_safetyarea / self.mapsize[0],
                    teammate_dis_white_safetyarea / self.mapsize[0],
                    teammate_dis_blue_safetyarea / (safety_area.radius +1 ),
                    teammate_dis_white_safetyarea / (safety_area.next_radius+1),
                    whether_teammate_run_in_circle_time,
                    whether_teammate_run_in_blue_circle,
                    whether_teammate_run_in_white_circle,
                ] + buff_list
                # teammate_info.append(teammate_info_k + buff_list)
        teammate_info = torch.as_tensor(teammate_info)

        teammate_info = {
            "character": teammate_info[:, 0].long(),
            'teammate_team_id': teammate_info[:, 1].long(),
            'teammate_pos_x': teammate_info[:, 2].float(),
            'teammate_pos_y': teammate_info[:, 3].float(),
            'teammate_pos_z': teammate_info[:, 4].float(),
            'teammate_rotation_x': teammate_info[:, 5].float(),
            'teammate_rotation_y': teammate_info[:, 6].float(),
            'teammate_rotation_z': teammate_info[:, 7].float(),
            'teammate_size_x': teammate_info[:, 8].float(),
            'teammate_size_y': teammate_info[:, 9].float(),
            'teammate_size_z': teammate_info[:, 10].float(),
            'teammate_speed_x': teammate_info[:, 11].float(),
            'teammate_speed_y': teammate_info[:, 12].float(),
            'teammate_speed_z': teammate_info[:, 13].float(),
            'teammate_scalar_speed': teammate_info[:, 14].float() / self.speed_norm,
            'teammate_hp': teammate_info[:, 15].float(),
            'teammate_neardeath_breath': teammate_info[:, 16].float(),
            'teammate_oxygen': teammate_info[:, 17].float(),
            'teammate_peek': teammate_info[:, 18].long(),
            'teammate_alive': teammate_info[:, 19].long(),
            'teammate_bodystate': teammate_info[:, 20].long(),
            'teammate_camera_position_x': teammate_info[:, 21].float(),
            'teammate_camera_position_y': teammate_info[:, 22].float(),
            'teammate_camera_position_z': teammate_info[:, 23].float(),
            'teammate_camera_rotation_x': teammate_info[:, 24].float(),
            'teammate_camera_rotation_y': teammate_info[:, 25].float(),
            'teammate_camera_rotation_z': teammate_info[:, 26].float(),
            'teammate_sin_rotation_x': teammate_info[:, 27].float(),
            'teammate_cos_rotation_x': teammate_info[:, 28].float(),
            'teammate_sin_rotation_y': teammate_info[:, 29].float(),
            'teammate_cos_rotation_y': teammate_info[:, 30].float(),
            'teammate_sin_rotation_z': teammate_info[:, 31].float(),
            'teammate_cos_rotation_z': teammate_info[:, 32].float(),
            'teammate_is_switching': teammate_info[:, 33].long(),
            'teammate_is_pose_changing': teammate_info[:, 34].long(),
            'teammate_is_running': teammate_info[:, 35].long(),
            'teammate_is_aiming': teammate_info[:, 36].long(),
            'teammate_is_firing': teammate_info[:, 37].long(),
            'teammate_is_holding': teammate_info[:, 38].long(),
            'teammate_is_falling': teammate_info[:, 39].long(),
            'teammate_is_picking': teammate_info[:, 40].long(),
            'teammate_player_vec_x': teammate_info[:, 41].float(),
            'teammate_player_vec_y': teammate_info[:, 42].float(),
            'teammate_player_vec_z': teammate_info[:, 43].float(),
            'teammate_player_dis': teammate_info[:, 44].float() ,
            'teammate_can_see_me': teammate_info[:, 45].long(),
            'teammate_sin_camera_rotation_x': teammate_info[:, 46].float(),
            'teammate_cos_camera_rotation_x': teammate_info[:, 47].float(),
            'teammate_sin_camera_rotation_y': teammate_info[:, 48].float(),
            'teammate_cos_camera_rotation_y': teammate_info[:, 49].float(),
            'teammate_sin_camera_rotation_z': teammate_info[:, 50].float(),
            'teammate_cos_camera_rotation_z': teammate_info[:, 51].float(),
            'teammate_in_blue_safetyarea': teammate_info[:, 52].long(),
            'teammate_in_white_safetyarea': teammate_info[:, 53].long(),
            'teammate_vec_blue_safetyarea_x': teammate_info[:, 54].float(),
            'teammate_vec_blue_safetyarea_y': teammate_info[:, 55].float(),
            'teammate_vec_white_safetyarea_x': teammate_info[:, 56].float(),
            'teammate_vec_white_safetyarea_y': teammate_info[:, 57].float(),
            'teammate_dis_blue_safetyarea_map': teammate_info[:, 58].float(),
            'teammate_dis_white_safetyarea_map': teammate_info[:, 59].float(),
            'teammate_dis_blue_safetyarea_radius': teammate_info[:, 60].float(),
            'teammate_dis_white_safetyarea_radius': teammate_info[:, 61].float(),
            'whether_teammate_run_in_circle_time': (teammate_info[:, 62].float() / self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'whether_teammate_run_in_blue_circle': (teammate_info[:, 63].float() / self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'whether_teammate_run_in_white_circle': (teammate_info[:, 64].float() / self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'teammate_buff_1': teammate_info[:, 65].long() ,
            'teammate_buff_2': teammate_info[:, 66].long() ,
            'teammate_buff_3': teammate_info[:, 67].long() ,
        }
        

        return teammate_info,teammate_alive_state,wait2saved_dist,top2bottom_teammate_position,pitch_teammate_position,yaw_teammate_position

    def transform_enemy_info(self, info, state, safety_area, state_info, own_player_state,search_target_player_id=None):
        #### enemy info
        self.nearest_enemy_dist = np.inf
        self.nearest_enemy_id = None
        timestap = state.timestamp - self.game_start_time
        safety_area_pos_x = safety_area.center.x 
        safety_area_pos_y = safety_area.center.y  
        safety_area_radius = safety_area.radius  
        safety_area_next_pos_x = safety_area.next_center.x  
        safety_area_next_pos_y = safety_area.next_center.y  
        safety_area_next_radius = safety_area.next_radius  
        safety_area_rest_time = safety_area.total_time - safety_area.time  # rest_time
        old_enemy_time_dict = {}
        # 统计有多少玩家活着
        self.alive_player_nums = 0
        for player_id, player_item in state_info.items():
            teammate_see_flag = False
            if info["alive_players"][player_id]:
                self.alive_player_nums += 1
            if (player_id == self.id or info['player_state'][
                player_id].state.team_id == own_player_state.state.team_id):
                continue
            elif (player_id not in own_player_state.visble_player_ids and player_id in self.ENEMY_SEE_INFO.keys()):
                # 当前不可见，且历史见过
                # TODO 未进一步处理，要复用上次信息！
                since_last_see_time = timestap - self.ENEMY_SEE_INFO[player_id]
                old_enemy_time_dict[player_id] = since_last_see_time
                continue
            elif (player_id not in own_player_state.visble_player_ids and player_id not in self.ENEMY_SEE_INFO.keys()):
                # 当前不可见，且历史没见过
                continue_flag = True
                for teammate_id in self.teammate_dict[self.id]:
                    if player_id in info['player_state'][teammate_id].visble_player_ids:
                        continue_flag = False
                        teammate_see_flag = True
                        break
                if continue_flag:
                    continue
                # 过滤未活着的, 可能有可见的盒子
            elif not info["alive_players"][player_id]:
                continue
            else:
                # 当前可见的敌人
                pass

            
            self.not_visble_enemy_time = state.timestamp  # 看见人以后更新
            
            since_last_see_time = 0  ###这次见到了
            self.ENEMY_SEE_INFO[player_id] = timestap
                
            enemy_character = all_characters[player_item.state.actor_id].id - 1100
            enemy_pos = player_item.state.position
            enemy_pos_x = player_item.state.position.x
            enemy_pos_y = player_item.state.position.y
            enemy_pos_z = player_item.state.position.z
            enemy_rotation_x = player_item.state.rotation.x  
            enemy_rotation_x_sin = math.sin(enemy_rotation_x*math.pi/180)
            enemy_rotation_x_cos = math.cos(enemy_rotation_x*math.pi/180)
            enemy_rotation_y = player_item.state.rotation.y  
            enemy_rotation_y_sin = math.sin(enemy_rotation_y*math.pi/180)
            enemy_rotation_y_cos = math.cos(enemy_rotation_y*math.pi/180)
            enemy_rotation_z = player_item.state.rotation.z 
            enemy_rotation_z_sin = math.sin(enemy_rotation_z*math.pi/180)
            enemy_rotation_z_cos = math.cos(enemy_rotation_z*math.pi/180)
            enemy_size_x = player_item.state.size.x  
            enemy_size_y = player_item.state.size.y  
            enemy_size_z = player_item.state.size.z
            enemy_speed_x = player_item.state.speed.x  
            enemy_speed_y = player_item.state.speed.y  
            enemy_speed_z = player_item.state.speed.z
            enemy_scalar_speed = min(math.sqrt( enemy_speed_x**2 + enemy_speed_y**2 + enemy_speed_z**2 ), 1000)
            enemy_hp = player_item.state.hp 
            enemy_neardeath_breath = player_item.state.neardeath_breath 
            enemy_oxygen = player_item.state.oxygen  
            enemy_peek = player_item.state.peek_type  
            enemy_alive = player_item.state.alive_state  
            enemy_bodystate = player_item.state.body_state  

            own_position = own_player_state.state.position
            enemy_relative_me_pos_x = enemy_pos.x - own_position.x
            enemy_relative_me_pos_y = enemy_pos.y - own_position.y
            enemy_relative_me_pos_z = enemy_pos.z - own_position.z
            enemy_distance = math.sqrt(enemy_relative_me_pos_x ** 2 + enemy_relative_me_pos_y ** 2 + enemy_relative_me_pos_z ** 2)
      
            if self.nearest_enemy_dist > enemy_distance and not (teammate_see_flag):
                self.nearest_enemy_dist = enemy_distance
                self.nearest_enemy_id = player_id
                
            enemy_team_id = player_item.state.team_id


            try:
                enemy_see_me = self.id in state_info[player_id].visble_player_ids
            except:
                enemy_see_me = False
            enemy_relative_blue_safetyarea_x,enemy_relative_blue_safetyarea_y, _ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_pos_x, safety_area_pos_y,0 ) 
            enemy_relative_white_safetyarea_x,enemy_relative_white_safetyarea_y,_ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_next_pos_x, safety_area_next_pos_y, 0) 
            enemy_distance_blue_safetyarea = math.sqrt(enemy_relative_blue_safetyarea_x**2+enemy_relative_blue_safetyarea_y**2)
            enemy_distance_white_safetyarea = math.sqrt(enemy_relative_white_safetyarea_x**2+enemy_relative_white_safetyarea_y**2)
            enemy_in_blue_safetyarea = enemy_distance_blue_safetyarea < safety_area_radius                                        
            enemy_in_white_safetyarea = enemy_distance_white_safetyarea < safety_area_next_radius
            whether_enemy_run_in_blue_circle_time = enemy_distance_blue_safetyarea / (self.expect_speed)-safety_area_rest_time
            whether_enemy_run_in_blue_circle = enemy_distance_blue_safetyarea / (self.expect_speed)
            whether_enemy_run_in_white_circle = enemy_distance_white_safetyarea / (self.expect_speed)

            hold_gun = 1 if len(player_item.weapon.player_weapon) > 0 else 0
            
            enemy_buff_list = [0] * 3
            for buff_idx,i in enumerate(player_item.state.buff):
                if buff_idx >= 3:
                    break
                enemy_buff_list[buff_idx] = i+1

            enemy_info = [enemy_distance,  ##0
                               enemy_team_id,
                               enemy_pos_x,
                               enemy_pos_y,
                               enemy_pos_z,
                               enemy_rotation_x,
                               enemy_rotation_y,
                               enemy_rotation_z,
                               enemy_rotation_x_sin,
                               enemy_rotation_y_sin,
                               enemy_rotation_z_sin,   ##10
                               enemy_rotation_x_cos,
                               enemy_rotation_y_cos,
                               enemy_rotation_z_cos,
                               enemy_size_x,
                               enemy_size_y,
                               enemy_size_z,
                               enemy_speed_x,
                               enemy_speed_y,
                               enemy_speed_z,
                               enemy_scalar_speed,    ##20
                               enemy_hp,
                               enemy_neardeath_breath,
                               enemy_oxygen,
                               enemy_peek,
                               enemy_alive,
                               enemy_bodystate,
                               enemy_relative_me_pos_x,
                               enemy_relative_me_pos_y,
                               enemy_relative_me_pos_z,                               
                               enemy_character,        ##30               
                               enemy_see_me,                               
                               enemy_relative_blue_safetyarea_x,
                               enemy_relative_blue_safetyarea_y,
                               enemy_relative_white_safetyarea_x,
                               enemy_relative_white_safetyarea_y,
                               enemy_distance_blue_safetyarea,
                               enemy_distance_white_safetyarea,
                               enemy_in_blue_safetyarea,
                               enemy_in_white_safetyarea,
                               whether_enemy_run_in_blue_circle_time,   ##40
                               whether_enemy_run_in_blue_circle,
                               whether_enemy_run_in_white_circle,
                               hold_gun,
                               since_last_see_time,
                               ] + enemy_buff_list


            if player_id not in self.enemy_id_queue:
                if len(self.enemy_id_queue)<self.max_player_num:
                    self.enemy_id_queue.append(player_id) ##新遇到的enemy，且队列未满
                    self.enemy_info_queue.append(enemy_info)
                else:                                           ###注意：这里是根据“多久没见过”来移除时间最长的对手信息
                    # longest_time_enemy_index = self.enemy_info_queue.index(max(self.enemy_info_queue[0][-1]))  
                    self.enemy_info_queue.remove(self.enemy_info_queue[0])
                    self.enemy_id_queue.remove(self.enemy_id_queue[0])
                    self.enemy_id_queue.append(player_id)  ##新遇到的enemy，但队列已满
                    self.enemy_info_queue.append(enemy_info)
            else:
                enemy_index = self.enemy_id_queue.index(player_id)
                # self.enemy_info_queue[enemy_index] = enemy_list

                self.enemy_id_queue.remove(self.enemy_id_queue[enemy_index])
                self.enemy_info_queue.remove(self.enemy_info_queue[enemy_index])
                self.enemy_id_queue.append(player_id)   
                self.enemy_info_queue.append(enemy_info)
        # 更新已经见过enemy的时间
        for player_id, since_last_see_time in old_enemy_time_dict.items():
            if player_id in self.enemy_id_queue:
                enemy_index = self.enemy_id_queue.index(player_id)
                self.enemy_info_queue[enemy_index][44] = since_last_see_time/(self.MAX_GAME_TIME*1000)
                enemy_relative_me_pos_x =  self.enemy_info_queue[enemy_index][2] - own_player_state.state.position.x
                enemy_relative_me_pos_y =  self.enemy_info_queue[enemy_index][3] - own_player_state.state.position.y
                enemy_relative_me_pos_z =  self.enemy_info_queue[enemy_index][4] - own_player_state.state.position.z
                distance = math.sqrt(enemy_relative_me_pos_x ** 2 + enemy_relative_me_pos_y ** 2 + enemy_relative_me_pos_z ** 2)
                self.enemy_info_queue[enemy_index][0] = distance
                self.enemy_info_queue[enemy_index][27] = enemy_relative_me_pos_x
                self.enemy_info_queue[enemy_index][28] = enemy_relative_me_pos_y
                self.enemy_info_queue[enemy_index][29] = enemy_relative_me_pos_z
                if not info["alive_players"][player_id]:
                    self.enemy_info_queue[enemy_index][21] = 0
                    self.enemy_info_queue[enemy_index][22] = 0
                    self.enemy_info_queue[enemy_index][25] = 2



        sorted_enemy_list = sorted(self.enemy_info_queue)

        # 为search做准备
        if search_target_player_id is not None:
            search_target = state_info[search_target_player_id]
            own_position = own_player_state.state.position
            enemy_relative_me_pos_x = search_target.state.position.x - own_position.x
            enemy_relative_me_pos_y = search_target.state.position.y - own_position.y
            enemy_relative_me_pos_z = search_target.state.position.z - own_position.z
            enemy_distance = math.sqrt(enemy_relative_me_pos_x ** 2 + enemy_relative_me_pos_y ** 2 + enemy_relative_me_pos_z ** 2)

            if enemy_distance >= 15000 or (search_target_player_id not in self.enemy_id_queue):
                enemy_info = self.get_search_target_info(search_target,own_player_state,state_info,search_target_player_id,\
                                                        safety_area_pos_x,safety_area_pos_y,safety_area_next_pos_x,safety_area_next_pos_y,\
                                                        safety_area_radius,safety_area_next_radius,safety_area_rest_time)

                sorted_enemy_list.append(enemy_info)
        

        all_enemy_item = torch.as_tensor(sorted_enemy_list).reshape(-1, 48)  # in case sorted_supply_item_list is empty
        enemy_item_num = len(all_enemy_item)
        enemy_item_padding_num = self.max_enemy_num - len(all_enemy_item)
        
        all_enemy_item = torch.nn.functional.pad(all_enemy_item, (0, 0, 0, enemy_item_padding_num), 'constant', 0)
        # all_enemy_item = torch.as_tensor(enemy_list).reshape(-1, 5)
        enemy_item_info = {
            'distance':all_enemy_item[:,0].float() / self.mapsize[0],              
            'team_id':all_enemy_item[:,1].long(),  ##TODO, binary 7, learner
            'pos_x':all_enemy_item[:,2].float() / self.mapsize[0],
            'pos_y':all_enemy_item[:,3].float() / self.mapsize[1],
            'pos_z':all_enemy_item[:,4].float() / self.mapsize[2],
            'rotation_x':all_enemy_item[:,5].float() / self.rotation_norm,
            'rotation_y':all_enemy_item[:,6].float() / self.rotation_norm,
            'rotation_z':all_enemy_item[:,7].float() / self.rotation_norm,
            'rotation_sin_x':all_enemy_item[:,8].float(),
            'rotation_sin_y':all_enemy_item[:,9].float(),
            'rotation_sin_z':all_enemy_item[:,10].float(),
            'rotation_cos_x':all_enemy_item[:,11].float(),
            'rotation_cos_y':all_enemy_item[:,12].float(),
            'rotation_cos_z':all_enemy_item[:,13].float(),
            'size_x':all_enemy_item[:,14].float() / self.size_xy_norm,
            'size_y':all_enemy_item[:,15].float() / self.size_xy_norm,
            'size_z':all_enemy_item[:,16].float() / self.size_z_norm,
            'speed_x':all_enemy_item[:,17].float() / self.speed_norm,
            'speed_y':all_enemy_item[:,18].float() / self.speed_norm,
            'speed_z':all_enemy_item[:,19].float() / self.speed_norm,
            'scalar_speed':all_enemy_item[:,20].float() / self.speed_norm,
            'hp':all_enemy_item[:,21].float() / self.hp_norm,
            'neardeath_breath':all_enemy_item[:,22].float() / self.hp_norm,                                    
            'oxygen':all_enemy_item[:,23].float() / self.oxygen_norm,
            'peek':all_enemy_item[:,24].long(),  ##onehot 3,
            'alive':all_enemy_item[:,25].long(),   ##onehot 3
            'bodystate':all_enemy_item[:,26].long(),  ##onehot 8
            'relative_pos_x':all_enemy_item[:,27].float() / (all_enemy_item[:,0] +1e-9),
            'relative_pos_y':all_enemy_item[:,28].float() / (all_enemy_item[:,0] +1e-9),  
            'relative_pos_z':all_enemy_item[:,29].float() / (all_enemy_item[:,0] +1e-9), 
            'character':all_enemy_item[:,30].long(),       ### binary 5                 
            'enemy_see_me':all_enemy_item[:,31].long(),  ###TODO
            'enemy_relative_blue_safetyarea_x':all_enemy_item[:,32].float()/ self.mapsize[0],
            'enemy_relative_blue_safetyarea_y':all_enemy_item[:,33].float()/ self.mapsize[1],  ##TODO
            'enemy_relative_white_safetyarea_x':all_enemy_item[:,34].float()/ self.mapsize[0],
            'enemy_relative_white_safetyarea_y':all_enemy_item[:,35].float()/ self.mapsize[1],
            'enemy_distance_blue_safetyarea':all_enemy_item[:,36].float()/ self.mapsize[0],
            'enemy_distance_blue_safetyarea_relative':all_enemy_item[:,36].float()/(safety_area_radius + 1),
            'enemy_distance_white_safetyarea':all_enemy_item[:,37].float()/ self.mapsize[0],
            'enemy_distance_white_safetyarea_relative':all_enemy_item[:,37].float()/(safety_area_next_radius + 1),
            'enemy_in_blue_safetyarea':all_enemy_item[:,38].long(),
            'enemy_in_white_safetyarea':all_enemy_item[:,39].long(),
            'whether_enemy_run_in_blue_circle_time':(all_enemy_item[:,40].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'whether_enemy_run_in_blue_circle':(all_enemy_item[:,41].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'whether_enemy_run_in_white_circle':(all_enemy_item[:,42].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),   
            "hold_gun": all_enemy_item[:, 43].long() ,               
            'since_last_see_time':all_enemy_item[:,44].float()/self.MAX_GAME_TIME,  
            
            'enemy_buff_1': all_enemy_item[:, 45].long() ,
            'enemy_buff_2': all_enemy_item[:, 46].long() ,
            'enemy_buff_3': all_enemy_item[:, 47].long() ,
             
            'enemy_item_num': torch.tensor(enemy_item_num, dtype=torch.long),
        }
        return enemy_item_info
    
    def get_search_target_info(self,player_item,own_player_state,state_info,search_target_player_id,\
                               safety_area_pos_x,safety_area_pos_y,safety_area_next_pos_x,safety_area_next_pos_y,\
                                safety_area_radius,safety_area_next_radius,safety_area_rest_time):

                
        since_last_see_time = 0  ###这次见到了
    
        enemy_character = all_characters[player_item.state.actor_id].id - 1100
        enemy_pos = player_item.state.position
        enemy_pos_x = player_item.state.position.x
        enemy_pos_y = player_item.state.position.y
        enemy_pos_z = player_item.state.position.z
        enemy_rotation_x = player_item.state.rotation.x  
        enemy_rotation_x_sin = math.sin(enemy_rotation_x*math.pi/180)
        enemy_rotation_x_cos = math.cos(enemy_rotation_x*math.pi/180)
        enemy_rotation_y = player_item.state.rotation.y  
        enemy_rotation_y_sin = math.sin(enemy_rotation_y*math.pi/180)
        enemy_rotation_y_cos = math.cos(enemy_rotation_y*math.pi/180)
        enemy_rotation_z = player_item.state.rotation.z 
        enemy_rotation_z_sin = math.sin(enemy_rotation_z*math.pi/180)
        enemy_rotation_z_cos = math.cos(enemy_rotation_z*math.pi/180)
        enemy_size_x = player_item.state.size.x  
        enemy_size_y = player_item.state.size.y  
        enemy_size_z = player_item.state.size.z
        enemy_speed_x = player_item.state.speed.x  
        enemy_speed_y = player_item.state.speed.y  
        enemy_speed_z = player_item.state.speed.z
        enemy_scalar_speed = min(math.sqrt( enemy_speed_x**2 + enemy_speed_y**2 + enemy_speed_z**2 ), 1000)
        enemy_hp = player_item.state.hp 
        enemy_neardeath_breath = player_item.state.neardeath_breath 
        enemy_oxygen = player_item.state.oxygen  
        enemy_peek = player_item.state.peek_type  
        enemy_alive = player_item.state.alive_state  
        enemy_bodystate = player_item.state.body_state  

        own_position = own_player_state.state.position
        enemy_relative_me_pos_x = enemy_pos.x - own_position.x
        enemy_relative_me_pos_y = enemy_pos.y - own_position.y
        enemy_relative_me_pos_z = enemy_pos.z - own_position.z
        enemy_distance = math.sqrt(enemy_relative_me_pos_x ** 2 + enemy_relative_me_pos_y ** 2 + enemy_relative_me_pos_z ** 2)


            
        enemy_team_id = player_item.state.team_id


        try:
            enemy_see_me = self.id in state_info[search_target_player_id].visble_player_ids
        except:
            enemy_see_me = False
        enemy_relative_blue_safetyarea_x,enemy_relative_blue_safetyarea_y, _ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_pos_x, safety_area_pos_y,0 ) 
        enemy_relative_white_safetyarea_x,enemy_relative_white_safetyarea_y,_ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_next_pos_x, safety_area_next_pos_y, 0) 
        enemy_distance_blue_safetyarea = math.sqrt(enemy_relative_blue_safetyarea_x**2+enemy_relative_blue_safetyarea_y**2)
        enemy_distance_white_safetyarea = math.sqrt(enemy_relative_white_safetyarea_x**2+enemy_relative_white_safetyarea_y**2)
        enemy_in_blue_safetyarea = enemy_distance_blue_safetyarea < safety_area_radius                                        
        enemy_in_white_safetyarea = enemy_distance_white_safetyarea < safety_area_next_radius
        whether_enemy_run_in_blue_circle_time = enemy_distance_blue_safetyarea / (self.expect_speed)-safety_area_rest_time
        whether_enemy_run_in_blue_circle = enemy_distance_blue_safetyarea / (self.expect_speed)
        whether_enemy_run_in_white_circle = enemy_distance_white_safetyarea / (self.expect_speed)

        hold_gun = 1 if len(player_item.weapon.player_weapon) > 0 else 0
        
        enemy_buff_list = [0] * 3
        for buff_idx,i in enumerate(player_item.state.buff):
            if buff_idx >= 3:
                break
            enemy_buff_list[buff_idx] = i+1

        enemy_info = [enemy_distance,  ##0
                        enemy_team_id,
                        enemy_pos_x,
                        enemy_pos_y,
                        enemy_pos_z,
                        enemy_rotation_x,
                        enemy_rotation_y,
                        enemy_rotation_z,
                        enemy_rotation_x_sin,
                        enemy_rotation_y_sin,
                        enemy_rotation_z_sin,   ##10
                        enemy_rotation_x_cos,
                        enemy_rotation_y_cos,
                        enemy_rotation_z_cos,
                        enemy_size_x,
                        enemy_size_y,
                        enemy_size_z,
                        enemy_speed_x,
                        enemy_speed_y,
                        enemy_speed_z,
                        enemy_scalar_speed,    ##20
                        enemy_hp,
                        enemy_neardeath_breath,
                        enemy_oxygen,
                        enemy_peek,
                        enemy_alive,
                        enemy_bodystate,
                        enemy_relative_me_pos_x,
                        enemy_relative_me_pos_y,
                        enemy_relative_me_pos_z,                               
                        enemy_character,        ##30               
                        enemy_see_me,                               
                        enemy_relative_blue_safetyarea_x,
                        enemy_relative_blue_safetyarea_y,
                        enemy_relative_white_safetyarea_x,
                        enemy_relative_white_safetyarea_y,
                        enemy_distance_blue_safetyarea,
                        enemy_distance_white_safetyarea,
                        enemy_in_blue_safetyarea,
                        enemy_in_white_safetyarea,
                        whether_enemy_run_in_blue_circle_time,   ##40
                        whether_enemy_run_in_blue_circle,
                        whether_enemy_run_in_white_circle,
                        hold_gun,
                        since_last_see_time,
                        ] + enemy_buff_list
        return enemy_info

    def transform_visible_enemy_info(self, info, state, safety_area, state_info, own_player_state):
        #### enemy info
        visible_enemy_info_queue = []
        visible_enemy_id_queue = []
        self.nearest_enemy_dist = np.inf
        self.nearest_enemy_id = None 
        safety_area_pos_x = safety_area.center.x 
        safety_area_pos_y = safety_area.center.y  
        safety_area_radius = safety_area.radius  
        safety_area_next_pos_x = safety_area.next_center.x  
        safety_area_next_pos_y = safety_area.next_center.y  
        safety_area_next_radius = safety_area.next_radius  
        safety_area_rest_time = safety_area.total_time - safety_area.time  # rest_time
        
        # 统计有多少玩家活着
        self.alive_player_nums = 0
        for player_id, player_item in state_info.items():
            if info["alive_players"][player_id]:
                self.alive_player_nums += 1
            if (player_id == self.id or info['player_state'][
                player_id].state.team_id == own_player_state.state.team_id):
                continue
            elif (player_id not in own_player_state.visble_player_ids):
                # 当前不可见 
                continue
            elif not info["alive_players"][player_id]:
                continue
            else:
                # 当前可见的敌人
                pass 
             
                
            enemy_character = all_characters[player_item.state.actor_id].id - 1100
            enemy_id = player_item.state.id
            enemy_pos = player_item.state.position
            enemy_pos_x = player_item.state.position.x
            enemy_pos_y = player_item.state.position.y
            enemy_pos_z = player_item.state.position.z
            enemy_rotation_x = player_item.state.rotation.x  
            enemy_rotation_x_sin = math.sin(enemy_rotation_x*math.pi/180)
            enemy_rotation_x_cos = math.cos(enemy_rotation_x*math.pi/180)
            enemy_rotation_y = player_item.state.rotation.y  
            enemy_rotation_y_sin = math.sin(enemy_rotation_y*math.pi/180)
            enemy_rotation_y_cos = math.cos(enemy_rotation_y*math.pi/180)
            enemy_rotation_z = player_item.state.rotation.z 
            enemy_rotation_z_sin = math.sin(enemy_rotation_z*math.pi/180)
            enemy_rotation_z_cos = math.cos(enemy_rotation_z*math.pi/180)
            enemy_size_x = player_item.state.size.x  
            enemy_size_y = player_item.state.size.y  
            enemy_size_z = player_item.state.size.z
            enemy_speed_x = player_item.state.speed.x  
            enemy_speed_y = player_item.state.speed.y  
            enemy_speed_z = player_item.state.speed.z
            enemy_scalar_speed = min(math.sqrt( enemy_speed_x**2 + enemy_speed_y**2 + enemy_speed_z**2 ), 1000)
            enemy_hp = player_item.state.hp 
            enemy_neardeath_breath = player_item.state.neardeath_breath 
            enemy_oxygen = player_item.state.oxygen  
            enemy_peek = player_item.state.peek_type  
            enemy_alive = player_item.state.alive_state  
            enemy_bodystate = player_item.state.body_state  

            own_position = own_player_state.state.position
            enemy_relative_me_pos_x = enemy_pos.x - own_position.x
            enemy_relative_me_pos_y = enemy_pos.y - own_position.y
            enemy_relative_me_pos_z = enemy_pos.z - own_position.z
            enemy_distance = math.sqrt(enemy_relative_me_pos_x ** 2 + enemy_relative_me_pos_y ** 2 + enemy_relative_me_pos_z ** 2)
      
            if self.nearest_enemy_dist > enemy_distance:
                self.nearest_enemy_dist = enemy_distance
                self.nearest_enemy_id = player_id
                
            enemy_team_id = player_item.state.team_id


            try:
                enemy_see_me = self.id in state_info[player_id].visble_player_ids
            except:
                enemy_see_me = False
            enemy_relative_blue_safetyarea_x,enemy_relative_blue_safetyarea_y, _ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_pos_x, safety_area_pos_y,0 ) 
            enemy_relative_white_safetyarea_x,enemy_relative_white_safetyarea_y,_ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_next_pos_x, safety_area_next_pos_y, 0) 
            enemy_distance_blue_safetyarea = math.sqrt(enemy_relative_blue_safetyarea_x**2+enemy_relative_blue_safetyarea_y**2)
            enemy_distance_white_safetyarea = math.sqrt(enemy_relative_white_safetyarea_x**2+enemy_relative_white_safetyarea_y**2)
            enemy_in_blue_safetyarea = enemy_distance_blue_safetyarea < safety_area_radius                                        
            enemy_in_white_safetyarea = enemy_distance_white_safetyarea < safety_area_next_radius
            whether_enemy_run_in_blue_circle_time = enemy_distance_blue_safetyarea / (self.expect_speed)-safety_area_rest_time
            whether_enemy_run_in_blue_circle = enemy_distance_blue_safetyarea / (self.expect_speed)
            whether_enemy_run_in_white_circle = enemy_distance_white_safetyarea / (self.expect_speed)

            hold_gun = 1 if len(player_item.weapon.player_weapon) > 0 else 0
            
            enemy_buff_list = [0] * 3
            for buff_idx,i in enumerate(player_item.state.buff):
                if buff_idx >= 3:
                    break
                enemy_buff_list[buff_idx] = i+1

            enemy_info = [enemy_distance,  ##0
                               enemy_team_id,
                               enemy_pos_x,
                               enemy_pos_y,
                               enemy_pos_z,
                               enemy_rotation_x,
                               enemy_rotation_y,
                               enemy_rotation_z,
                               enemy_rotation_x_sin,
                               enemy_rotation_y_sin,
                               enemy_rotation_z_sin,   ##10
                               enemy_rotation_x_cos,
                               enemy_rotation_y_cos,
                               enemy_rotation_z_cos,
                               enemy_size_x,
                               enemy_size_y,
                               enemy_size_z,
                               enemy_speed_x,
                               enemy_speed_y,
                               enemy_speed_z,
                               enemy_scalar_speed,    ##20
                               enemy_hp,
                               enemy_neardeath_breath,
                               enemy_oxygen,
                               enemy_peek,
                               enemy_alive,
                               enemy_bodystate,
                               enemy_relative_me_pos_x,
                               enemy_relative_me_pos_y,
                               enemy_relative_me_pos_z,                               
                               enemy_character,        ##30               
                               enemy_see_me,                               
                               enemy_relative_blue_safetyarea_x,
                               enemy_relative_blue_safetyarea_y,
                               enemy_relative_white_safetyarea_x,
                               enemy_relative_white_safetyarea_y,
                               enemy_distance_blue_safetyarea,
                               enemy_distance_white_safetyarea,
                               enemy_in_blue_safetyarea,
                               enemy_in_white_safetyarea,
                               whether_enemy_run_in_blue_circle_time,   ##40
                               whether_enemy_run_in_blue_circle,
                               whether_enemy_run_in_white_circle,
                               hold_gun,
                               ] + enemy_buff_list 


 
            # if len(visible_enemy_id_queue)<self.max_visible_player_num:
            visible_enemy_id_queue.append([enemy_distance,player_id]) ##新遇到的enemy，且队列未满
            visible_enemy_info_queue.append(enemy_info)

 

        
        sorted_enemy_list = sorted(visible_enemy_info_queue)
        self.visible_enemy_id_queue = sorted(visible_enemy_id_queue)

        all_enemy_item = torch.as_tensor(sorted_enemy_list).reshape(-1, 47)  # in case sorted_supply_item_list is empty
        enemy_item_num = len(all_enemy_item)
        enemy_item_padding_num = self.max_visible_player_num - len(all_enemy_item)
        
        all_enemy_item = torch.nn.functional.pad(all_enemy_item, (0, 0, 0, enemy_item_padding_num), 'constant', 0)
        # all_enemy_item = torch.as_tensor(enemy_list).reshape(-1, 5)
        visible_enemy_item_info = {
            'distance':all_enemy_item[:,0].float() / self.mapsize[0],              
            'team_id':all_enemy_item[:,1].long(),  ##TODO, binary 7, learner
            'pos_x':all_enemy_item[:,2].float() / self.mapsize[0],
            'pos_y':all_enemy_item[:,3].float() / self.mapsize[1],
            'pos_z':all_enemy_item[:,4].float() / self.mapsize[2],
            'rotation_x':all_enemy_item[:,5].float() / self.rotation_norm,
            'rotation_y':all_enemy_item[:,6].float() / self.rotation_norm,
            'rotation_z':all_enemy_item[:,7].float() / self.rotation_norm,
            'rotation_sin_x':all_enemy_item[:,8].float(),
            'rotation_sin_y':all_enemy_item[:,9].float(),
            'rotation_sin_z':all_enemy_item[:,10].float(),
            'rotation_cos_x':all_enemy_item[:,11].float(),
            'rotation_cos_y':all_enemy_item[:,12].float(),
            'rotation_cos_z':all_enemy_item[:,13].float(),
            'size_x':all_enemy_item[:,14].float() / self.size_xy_norm,
            'size_y':all_enemy_item[:,15].float() / self.size_xy_norm,
            'size_z':all_enemy_item[:,16].float() / self.size_z_norm,
            'speed_x':all_enemy_item[:,17].float() / self.speed_norm,
            'speed_y':all_enemy_item[:,18].float() / self.speed_norm,
            'speed_z':all_enemy_item[:,19].float() / self.speed_norm,
            'scalar_speed':all_enemy_item[:,20].float() / self.speed_norm,
            'hp':all_enemy_item[:,21].float() / self.hp_norm,
            'neardeath_breath':all_enemy_item[:,22].float() / self.hp_norm,                                    
            'oxygen':all_enemy_item[:,23].float() / self.oxygen_norm,
            'peek':all_enemy_item[:,24].long(),  ##onehot 3,
            'alive':all_enemy_item[:,25].long(),   ##onehot 3
            'bodystate':all_enemy_item[:,26].long(),  ##onehot 8
            'relative_pos_x':all_enemy_item[:,27].float() / (all_enemy_item[:,0] +1e-9),
            'relative_pos_y':all_enemy_item[:,28].float() / (all_enemy_item[:,0] +1e-9),  
            'relative_pos_z':all_enemy_item[:,29].float() / (all_enemy_item[:,0] +1e-9), 
            'character':all_enemy_item[:,30].long(),       ### binary 5                 
            'enemy_see_me':all_enemy_item[:,31].long(),  ###TODO
            'enemy_relative_blue_safetyarea_x':all_enemy_item[:,32].float()/ self.mapsize[0],
            'enemy_relative_blue_safetyarea_y':all_enemy_item[:,33].float()/ self.mapsize[1],  ##TODO
            'enemy_relative_white_safetyarea_x':all_enemy_item[:,34].float()/ self.mapsize[0],
            'enemy_relative_white_safetyarea_y':all_enemy_item[:,35].float()/ self.mapsize[1],
            'enemy_distance_blue_safetyarea':all_enemy_item[:,36].float()/ self.mapsize[0],
            'enemy_distance_blue_safetyarea_relative':all_enemy_item[:,36].float()/(safety_area_radius + 1),
            'enemy_distance_white_safetyarea':all_enemy_item[:,37].float()/ self.mapsize[0],
            'enemy_distance_white_safetyarea_relative':all_enemy_item[:,37].float()/(safety_area_next_radius + 1),
            'enemy_in_blue_safetyarea':all_enemy_item[:,38].long(),
            'enemy_in_white_safetyarea':all_enemy_item[:,39].long(),
            'whether_enemy_run_in_blue_circle_time':(all_enemy_item[:,40].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'whether_enemy_run_in_blue_circle':(all_enemy_item[:,41].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'whether_enemy_run_in_white_circle':(all_enemy_item[:,42].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),                             
            "hold_gun": all_enemy_item[:, 43].long() ,   
            'enemy_buff_1': all_enemy_item[:, 44].long() ,
            'enemy_buff_2': all_enemy_item[:, 45].long() ,
            'enemy_buff_3': all_enemy_item[:, 46].long() ,             
            'enemy_item_num': torch.tensor(enemy_item_num, dtype=torch.long),
            'act_attention_shift': torch.tensor(self.cfg.agent.act_attention_shift, dtype=torch.float),
        }
        return visible_enemy_item_info



    def transform_backpack_info(self,info,item_id2item_category,bullet_cnt_dict,scalar_info):

        state_info = info['player_state']

        own_player_state = state_info[self.id]
        
        
        # ================
        # 背包 backpack @wj
        # ================
        """
        投掷物（手雷、闪光弹、烟雾弹的占用空间暂未提供）
        """
        self.recovery_hp_items_list = []
        backpack_items = own_player_state.backpack.backpack_item
        weapons_infos = [(p.category,p.slot_id,p.attachments) for p in own_player_state.weapon.player_weapon]

        backpack_item_list = []
        volume_total = self.bag_volumes["default"]
        volume_used = 0
        
        
        
        
        for backpack_item in backpack_items:
            if backpack_item.category not in self.ALL_SUPPLY_ITEMS.keys():
                continue
            # 更新背包物资信息
            item_idx = self.backpack_update_info["item"].index(backpack_item.category)
            self.backpack_update_info["count"][item_idx] = backpack_item.count
            
            if backpack_item.category in RecoveryHP:
                self.recovery_hp_items_list.append(backpack_item)

            ##-------- 背包物品类型细分表示
            backpack_item_class = self.ALL_SUPPLY_ITEMS[backpack_item.category]
            backpack_item_type = backpack_item_class.type.value # 大类  1~10可限制为
            if backpack_item_type not in self.backpack_type_buffers:
                self.backpack_type_buffers.append(backpack_item_type)
            
            backpack_item_subtype = backpack_item_class.subtype # 细分小类 1~10可限制为
            backpack_item_sub_id = backpack_item_class.sub_id # 细致id 1~10的数字
            
            # # 背包物品局内id 没有实际意义 不使用
            # backpack_item_id = backpack_item.id
            item_id2item_category[backpack_item.id] = backpack_item.category
            
            # 物品数量
            backpack_item_count = backpack_item.count 
            
            # 物品占用空间
            backpack_item_size = backpack_item_class.size * backpack_item_count
            
            # 已使用容量
            if (not backpack_item.in_use):
                volume_used += backpack_item_class.size * backpack_item_count
            
            # 最大容量
            if backpack_item.category in self.bag_volumes.keys():
                volume_total = self.bag_volumes[backpack_item.category]
                
            # ----是否为配件(或子弹)且可装配
            match_info = [0] * self.max_slot_nums
            used_in_slot = 0
            # 配件匹配
            if backpack_item.category in self.AttachmentCategorys:
                for category,slot_id,attachments in weapons_infos:
                    if self.AttachmentUseble[self.GunCategorys.index(category)][
                        self.AttachmentCategorys.index(backpack_item.category)]:
                        match_info[slot_id] = 1
                    if backpack_item.id in attachments:
                        used_in_slot = slot_id
            # 子弹匹配
            elif backpack_item.category in self.bullet2gun.keys():
                bullet_cnt_dict[backpack_item.category] = backpack_item_count
                for category,slot_id,_ in weapons_infos:
                    if category in self.bullet2gun[backpack_item.category]:
                        match_info[slot_id] = 1
            
            backpack_item_list.append([backpack_item_type,
                                       backpack_item_subtype,
                                       backpack_item_sub_id,
                                       backpack_item_count,
                                       backpack_item_size,
                                       used_in_slot] + match_info)
   
        
        # padding
        all_backpack_item = torch.as_tensor(backpack_item_list).reshape(-1, 11)  # in case sorted_door_list is empty
        backpack_item_padding_num = self.max_backpack_item_num - len(all_backpack_item)
        backpack_item_num = len(all_backpack_item)
        all_backpack_item = torch.nn.functional.pad(all_backpack_item, (0, 0, 0, backpack_item_padding_num), 'constant', 0)
        
        backpack_item_info = {             
            'main_type': all_backpack_item[:, 0].long(),# 大类  1~10可限制为 bianry 5           
            'subtype': all_backpack_item[:, 1].long(),# 细分小类 1~10可限制为   bianry 5           
            'sub_id': all_backpack_item[:,2].long(),# 细致id 1~10的数字  bianry 5            
            'count': torch.clip(all_backpack_item[:,3],0,255).long(),# 可使用二进制进行编码为 < 256的数 bianry 8
            'size': (torch.clip(all_backpack_item[:,4]/self.backup_volume_norm,0,1.)).float(), # 物品占用背包容量 可除200   
            'used_in_slot' :all_backpack_item[:,5].long(),# 配件已被用在第几个武器槽 -- 使用长度为5的onehot编码
            'slot_0': all_backpack_item[:,6].long(), # 武器槽0是否可装备配件或子弹
            'slot_1': all_backpack_item[:,7].long(), # 武器槽1是否可装备配件或子弹
            'slot_2': all_backpack_item[:,8].long(), # 武器槽2是否可装备配件或子弹
            'slot_3': all_backpack_item[:,9].long(), # 武器槽3是否可装备配件或子弹
            'slot_4': all_backpack_item[:,10].long(), # 武器槽4是否可装备配件或子弹
            'backpack_item_num': torch.tensor(backpack_item_num, dtype=torch.long),# 使用transformer进行encoder时可以使用            
            # 'backpack_volume_used': torch.tensor(volume_used, dtype=torch.long),# 背包已使用量 可除200 
            
            # 放入self info 
            # 'backpack_volume_total': torch.tensor(volume_total/self.backup_volume_norm, dtype=torch.float),# 背包总量 可除200            
            # 'backpack_volume_rest': torch.tensor((volume_total - volume_used)/self.backup_volume_norm, dtype=torch.float),# 背包剩余量 可除200            
            # 'backpack_volume_percent': torch.tensor(volume_used/volume_total, dtype=torch.float),# 背包已使用占比 直接使用
        }
        
        scalar_info["backpack_volume_total"] = torch.tensor(volume_total/self.backup_volume_norm, dtype=torch.float)  # 背包总量 可除200   
        scalar_info["backpack_volume_rest"] = torch.tensor((volume_total - volume_used)/self.backup_volume_norm, dtype=torch.float)   # 背包剩余量 可除200           
        scalar_info["backpack_volume_percent"] = torch.tensor(volume_used/volume_total, dtype=torch.float)    # 背包已使用占比 直接使用
        
        return backpack_item_info,item_id2item_category,bullet_cnt_dict,scalar_info


    def transform_supply_info(self,info, own_position, safety_area, state,own_player_state, scalar_info):


        # ==========
        # supply_items
        # ==========
        supply_items = info['items']
        # print(f'supply_items length :{len(items)}')

        # 预先将supplys的位置信息进行储存，后续使用矩阵运算筛选出可见距离内的supply！
        if self.supplys_pos_matrix is None:
            self.supplys_pos_matrix = []
            for sup_id, sup in supply_items.items():
                self.supplys_categorys.append(sup_id)
                self.supplys_pos_matrix.append([sup.position.x,sup.position.y,sup.position.z])
            self.supplys_pos_matrix = torch.as_tensor(self.supplys_pos_matrix)
            self.supplys_categorys = torch.as_tensor(self.supplys_categorys)
            
        # 求个差集,追加supply
        supplys_buffers_old = self.supplys_categorys.tolist()
        supplys_buffers_new = list(supply_items.keys())
        supply_deltas = list(set(supplys_buffers_new).difference(set(supplys_buffers_old)))
        for sup_id in supply_deltas:
            supply_item = supply_items[sup_id]
            supply_item_pos = supply_item.position
            self.supplys_pos_matrix = torch.cat([self.supplys_pos_matrix,torch.tensor([[supply_item_pos.x,supply_item_pos.y,supply_item_pos.z]])],dim=0)
            self.supplys_categorys = torch.cat([self.supplys_categorys,torch.tensor([sup_id])],dim=0)
            
        # 找出范围内的supply
        visble_supplys = self.find_nearest_supplys([own_position.x,own_position.y,own_position.z])
        # self.nearest_supply = None
        

        
        supply_prioritys = []
        supply_mask_distance = np.inf
        for supply_item_id in visble_supplys:
            supply_item_id = supply_item_id.item()
            if supply_item_id not in supply_items.keys():
                continue
            supply_item = supply_items[supply_item_id]
            if supply_item.count == 0 or supply_item.attribute != 1: # attribute 1:初始状态，0:被捡
                continue
            if supply_item.category not in all_supply_items.keys():
                continue
            supply_item_pos = supply_item.position
            supply_item_category = supply_item.category
            supply_item_x = supply_item_pos.x
            supply_item_y = supply_item_pos.y
            supply_item_z = supply_item_pos.z
            
            # 检查是否更新supply的坐标
            sup_idx = torch.nonzero(torch.eq(self.supplys_categorys,supply_item_id))[0][0]
            pos_old = self.supplys_pos_matrix[sup_idx]
            if self.computer_distance(supply_item_x,supply_item_y,supply_item_z,pos_old[0],pos_old[1],pos_old[2]) > 5:
                self.supplys_pos_matrix[sup_idx] = torch.tensor([supply_item_x,supply_item_y,supply_item_z])
                if self.computer_distance(supply_item_x,supply_item_y,supply_item_z,own_position.x,own_position.y,own_position.z) > self.supply_visble_distance:
                    continue
                

            supply_item_rel_x = supply_item_pos.x - own_position.x
            supply_item_rel_y = supply_item_pos.y - own_position.y
            supply_item_rel_z = supply_item_pos.z - own_position.z
            
            supply_item_class = all_supply_items[supply_item_category]
            supply_item_type = supply_item_class.type
            supply_item_subtype = supply_item_class.subtype
            supply_item_sub_id = supply_item_class.sub_id
            supply_item_size = self.ALL_SUPPLY_ITEMS[supply_item.category].size
            supply_item_airdrop = supply_item_category == 0    ### TODO
            distance = math.sqrt(supply_item_rel_x ** 2 + supply_item_rel_y ** 2 + supply_item_rel_z ** 2)
            
            if supply_mask_distance > distance:
                supply_mask_distance = distance
            item_class = self.ALL_SUPPLY_ITEMS[supply_item.category]
            item_type = item_class.type.value # 大类  1~10可限制为
            
            supply_center_rel_x = supply_item_pos.x - safety_area.center.x 
            supply_center_rel_y = supply_item_pos.y - safety_area.center.y  
            supply2blue_dis = (supply_center_rel_x**2 + supply_center_rel_y**2)**0.5
            
            if self.target_supply_mode == "random":
                supply_prioritys.append([item_class.priority,-distance,supply_item_id])   
            else:
                if item_type not in self.backpack_type_buffers and supply2blue_dis < safety_area.radius:
                    supply_prioritys.append([item_class.priority,-distance,supply_item_id])            

            since_last_see_time = 0  ###这次见到了
            timestap = state.timestamp - self.game_start_time
            self.SUPPLY_SEE_INFO[supply_item_id] = timestap


            supply_info = ([
                distance,
                supply_item.count,
                supply_item.attribute,
                supply_item_x,
                supply_item_y,
                supply_item_z,
                supply_item_rel_x,
                supply_item_rel_y,
                supply_item_rel_z,
                supply_item_airdrop,
                supply_item_type,
                supply_item_subtype,
                supply_item_sub_id,  
                # supply_item_size,
                since_last_see_time              
            ])
            if supply_item_id not in self.supply_id_queue:
                if len(self.supply_id_queue)<self.max_supply_num:
                    self.supply_id_queue.append(supply_item_id)  ##新遇到的supply，且队列未满
                    self.supply_info_queue.append(supply_info)
                else:                                           ###注意：这里是根据“多久没见过”来移除时间最长的对手信息
                    # longest_time_supply_index = self.supply_info_queue.index(max(self.supply_info_queue[-1]))  
                    self.supply_info_queue.remove(self.supply_info_queue[0])
                    self.supply_id_queue.remove(self.supply_id_queue[0])
                    self.supply_id_queue.append(supply_item_id)  ##新遇到的supply，但队列已满
                    self.supply_info_queue.append(supply_info)
            else:
                supply_index = self.supply_id_queue.index(supply_item_id)
                # self.supply_info_queue[supply_index] = supply_item_list

                self.supply_info_queue.remove(self.supply_info_queue[supply_index])
                self.supply_id_queue.remove(self.supply_id_queue[supply_index])
                self.supply_id_queue.append(supply_item_id)  ##新遇到的supply，但队列已满
                self.supply_info_queue.append(supply_info)
        ## 更新未见到的supply的计时器
        for key in self.SUPPLY_SEE_INFO.keys():
            if key not in visble_supplys:
                # TODO 待確認
                if key in self.supply_id_queue:
                    index = self.supply_id_queue.index(key)
                    self.supply_info_queue[index][-1] = (state.timestamp - self.SUPPLY_SEE_INFO[key]) / (self.MAX_GAME_TIME * 1000)

        target_supply_feature = [0] * 7

        supply_update_flag = False
        # if self.nearest_supply is not None and self.nearest_supply[2]   in info['items'].keys():            
                
        #     supply_item = info['items'][self.nearest_supply[2]]
        #     supply_item_rel_x = supply_item.position.x - own_player_state.state.position.x 
        #     supply_item_rel_y = supply_item.position.y - own_player_state.state.position.y
        #     supply_item_rel_z = supply_item.position.z - own_player_state.state.position.z

            
        #     supply2me_dis = (supply_item_rel_x ** 2 + supply_item_rel_y**2 + supply_item_rel_z**2)**0.5
        #     target_supply_feature = [supply_item.position.x,supply_item.position.y,supply_item.position.z,\
        #                                supply_item_rel_x,supply_item_rel_y,supply_item_rel_z,supply2me_dis]
        #     self.nearest_supply[1] = -supply2me_dis
        #     if supply_item.attribute == 0 or self.approch_supply_step >= 20:
        #         supply_update_flag = True
        #         print("update_supply: supply is picked up or approch step exceed 25!")
        #     elif self.approch_supply_step > 0:
        #         self.approch_supply_step += 1
        #     elif supply2me_dis <= 200:
        #         self.agent_static_info["approch_target_cnt"] = self.agent_static_info["approch_target_cnt"] + 1
        #         print("approch supply firstly!!!!")
        #         self.approch_supply_step = 1
                
        #     supply_center_rel_x = supply_item.position.x - safety_area.center.x 
        #     supply_center_rel_y = supply_item.position.y - safety_area.center.y  
        #     supply2blue_dis = (supply_center_rel_x**2 + supply_center_rel_y**2)**0.5
        #     if supply2blue_dis > safety_area.radius:
        #         print("update_supply: supply is out of safety_area!")
        #         supply_update_flag = True        
        if self.nearest_supply is not None:
            if self.nearest_supply[2] not in info['items'].keys():
                supply_update_flag = True
            else:
                supply_item = info['items'][self.nearest_supply[2]]
                supply_item_rel_x = supply_item.position.x - own_player_state.state.position.x 
                supply_item_rel_y = supply_item.position.y - own_player_state.state.position.y
                supply_item_rel_z = supply_item.position.z - own_player_state.state.position.z

                
                supply2me_dis = (supply_item_rel_x ** 2 + supply_item_rel_y**2 + supply_item_rel_z**2)**0.5
                target_supply_feature = [supply_item.position.x,supply_item.position.y,supply_item.position.z,\
                                        supply_item_rel_x,supply_item_rel_y,supply_item_rel_z,supply2me_dis]
                self.nearest_supply[1] = -supply2me_dis
                if supply_item.attribute == 0 or self.approch_supply_step >= 20:
                    supply_update_flag = True
                    print("update_supply: supply is picked up or approch step exceed 25!")
                elif self.approch_supply_step > 0:
                    self.approch_supply_step += 1
                elif supply2me_dis <= 200:
                    self.agent_static_info["approch_target_cnt"] = self.agent_static_info["approch_target_cnt"] + 1
                    print("approch supply firstly!!!!")
                    self.approch_supply_step = 1
                    
                supply_center_rel_x = supply_item.position.x - safety_area.center.x 
                supply_center_rel_y = supply_item.position.y - safety_area.center.y  
                supply2blue_dis = (supply_center_rel_x**2 + supply_center_rel_y**2)**0.5
                if supply2blue_dis > safety_area.radius:
                    print("update_supply: supply is out of safety_area!")
                    supply_update_flag = True
        else:
            supply_update_flag = True
        if  supply_update_flag:
            self.approch_supply_step = 0
            if len(supply_prioritys) != 0:
                if self.target_supply_mode == "random":
                    self.nearest_supply = random.choice(supply_prioritys)
                else:
                    supply_prioritys = sorted(supply_prioritys,key=lambda x:(x[0],x[1]))
                    # self.nearest_supply = supply_prioritys[-1]
                    self.nearest_supply = supply_prioritys[-1]
                supply_item = info['items'][self.nearest_supply[2]]
                supply_item_rel_x = supply_item.position.x - own_player_state.state.position.x 
                supply_item_rel_y = supply_item.position.y - own_player_state.state.position.y
                supply_item_rel_z = supply_item.position.z - own_player_state.state.position.z
                
                supply2me_dis = (supply_item_rel_x ** 2 + supply_item_rel_y**2 + supply_item_rel_z**2)**0.5
                target_supply_feature = [supply_item.position.x,supply_item.position.y,supply_item.position.z,\
                                        supply_item_rel_x,supply_item_rel_y,supply_item_rel_z,supply2me_dis]
            else:
                self.nearest_supply = None
                target_supply_feature = [0] * 7

        scalar_info["target_x"] = torch.tensor(target_supply_feature[0] / self.mapsize[0], dtype=torch.float)
        scalar_info["target_y"] = torch.tensor(target_supply_feature[1] / self.mapsize[1], dtype=torch.float)
        scalar_info["target_z"] = torch.tensor(target_supply_feature[2] / self.mapsize[2], dtype=torch.float)
        scalar_info["target_x_rel"] = torch.tensor(target_supply_feature[3] / (target_supply_feature[6] + 1e-9), dtype=torch.float)
        scalar_info["target_y_rel"] = torch.tensor(target_supply_feature[4] / (target_supply_feature[6] + 1e-9), dtype=torch.float)
        scalar_info["target_z_rel"] = torch.tensor(target_supply_feature[5] / (target_supply_feature[6] + 1e-9), dtype=torch.float)
        scalar_info["target_distance"] = torch.tensor(target_supply_feature[6] / self.mapsize[0], dtype=torch.float)
                

        sorted_supply_list = sorted(self.supply_info_queue)
        all_supply_item = torch.as_tensor(sorted_supply_list).reshape(-1, 14)  # in case sorted_supply_item_list is empty
        supply_item_num = len(all_supply_item)
        supply_item_padding_num = self.max_supply_num - len(all_supply_item)
        
        all_supply_item = torch.nn.functional.pad(all_supply_item, (0, 0, 0, supply_item_padding_num), 'constant', 0)

        supply_item_info = {
            'distance': all_supply_item[:, 0].float() /self.supply_visble_distance,
            'quantity': all_supply_item[:, 1].float() / self.quantity_norm,   ### binary 8
            'attribute': all_supply_item[:, 2].long(),
            'pos_x': all_supply_item[:, 3].float()/self.mapsize[0],
            'pos_y': all_supply_item[:, 4].float()/self.mapsize[1],
            'pos_z': all_supply_item[:, 5].float()/self.mapsize[2],
            'relative_pos_x':all_supply_item[:, 6].float()/(all_supply_item[:, 0]+1e-9),
            'relative_pos_y':all_supply_item[:, 7].float()/(all_supply_item[:, 0]+1e-9),
            'relative_pos_z':all_supply_item[:, 8].float()/(all_supply_item[:, 0]+1e-9),
            # 'pitch': supply_item_pitch,
            # 'yaw': supply_item_yaw,
            'air_drop': all_supply_item[:, 9].long(),
            'main_type': all_supply_item[:, 10].long(),  ##binary 5
            'subtype': all_supply_item[:, 11].long(),   ##binary 5
            'sub_id': all_supply_item[:, 12].long(),    ##binary 5
            'size':(torch.clip(all_supply_item[:, 13].float()*all_supply_item[:, 1],0,250))/self.backup_volume_norm, ##TODO
            'supply_item_num': torch.tensor(supply_item_num, dtype=torch.long),
        }

        return supply_item_info,scalar_info,supply_mask_distance

    def transform_weapon_info(self,bullet_cnt_dict, own_player_state, item_id2item_category):
                # ===============
        # 武器 weapon @wj
        # ===============
        self.reward_info["bullet_sum"] = len(bullet_cnt_dict)
        max_weapon_remain_reloading = 0
        activate_slot_id = 0
        self.activate_category = 0
        activate_slot_bullet = 0
        activate_slot_capacity = np.inf
        weapon_mask = [1,0,0]
        
        player_weapons = own_player_state.weapon.player_weapon
        
        player_weapon_list = [[0 for _ in range(27)] for num in range(self.max_player_weapon_num)]
        
        player_weapon_num = 0
        
        for player_weapon in player_weapons:
            if player_weapon.slot_id <=2:
                weapon_mask[player_weapon.slot_id] = 1
            player_weapon_num += 1
            # 是否为当前激活武器
            player_weapon_is_active = int(player_weapon.slot_id == own_player_state.state.active_weapon_slot)
            
            if player_weapon_is_active:
                self.activate_category = player_weapon.category
                activate_slot_id = player_weapon.slot_id
                activate_slot_bullet = player_weapon.bullet
                activate_slot_capacity = player_weapon.capacity
            # # 武器局内id 莫有啥用
            # player_weapon_id = player_weapon.id
            
            ##-------- 武器编号细分表示        
            player_weapon_class = self.ALL_SUPPLY_ITEMS[player_weapon.category]
            player_weapon_type = player_weapon_class.type.value # 大类  1~10可限制为
            player_weapon_subtype = player_weapon_class.subtype # 细分小类 1~10可限制为
            player_weapon_sub_id = player_weapon_class.sub_id # 细致id 1~10的数字
            
            # 武器保有子弹数
            player_weapon_current_bullet = player_weapon.bullet
            
            # 武器备弹数
            player_weapon_rest_bullet = bullet_cnt_dict.get(self.gun2bullet[player_weapon.category],0)
            
            # 武器弹夹容量
            player_weapon_capacity = player_weapon.capacity

            
            
            # 武器保有子弹比率
            player_weapon_bullet_percent =  player_weapon_current_bullet / (player_weapon_capacity + 1e-7) 
            
            # 武器内子弹占备弹比率 可弃用
            # player_weapon_bullet2backpack_percent =  player_weapon_rest_bullet / player_weapon_capacity 
            
            # 剩余填充时间
            player_weapon_remain_reloading = player_weapon.remain_reloading
            max_weapon_remain_reloading = max(max_weapon_remain_reloading,player_weapon_remain_reloading)
            
            # # -- 武器配件表 --- 不一定使用
            # # 武器可用哪些配件
            # player_weapon_can_use_attachments = self.AttachmentUseble[self.GunCategorys.index(player_weapon.category)]
            # # 武器已用配件
            # player_weapon_used_attachments = [0 for _ in range(len(player_weapon_can_use_attachments))]  
            # for attachment_id in player_weapon.attachments:
            #     player_weapon_used_attachments[self.AttachmentCategorys.index(item_id2item_category[attachment_id])] = 1
            
            # 武器已用配件编号
            # -----------
            # 枪口、握把、枪托、弹夹、瞄具
            # -----------
            player_weapon_attachments_locs = [0] * (3 * self.max_attachment_nums)
            
            for attachment_id in player_weapon.attachments:
                attachment_class = self.ALL_SUPPLY_ITEMS[item_id2item_category[attachment_id]]
                attachment_type = attachment_class.type.value # 大类  1~10可限制为
                attachment_subtype = attachment_class.subtype # 细分小类 1~10可限制为
                attachment_sub_id = attachment_class.sub_id # 细致id 1~10的数字
                if item_id2item_category[attachment_id] in self.AttachmentsLocs["Muzzle"]:
                    player_weapon_attachments_locs[0:3] = [attachment_type,attachment_subtype,attachment_sub_id]
                elif item_id2item_category[attachment_id] in self.AttachmentsLocs["grip"]:
                    player_weapon_attachments_locs[3:6] = [attachment_type,attachment_subtype,attachment_sub_id]
                elif item_id2item_category[attachment_id] in self.AttachmentsLocs["butt"]:
                    player_weapon_attachments_locs[6:9] = [attachment_type,attachment_subtype,attachment_sub_id]
                elif item_id2item_category[attachment_id] in self.AttachmentsLocs["clip"]:
                    player_weapon_attachments_locs[9:12] = [attachment_type,attachment_subtype,attachment_sub_id]
                elif item_id2item_category[attachment_id] in self.AttachmentsLocs["sight"]:
                    player_weapon_attachments_locs[12:15] = [attachment_type,attachment_subtype,attachment_sub_id]
            
            # 武器使用子弹编号
            
            bullet_class = self.ALL_SUPPLY_ITEMS[self.gun2bullet[player_weapon.category]]
            bullet_type = bullet_class.type.value # 大类  1~10可限制为
            bullet_subtype = bullet_class.subtype # 细分小类 1~10可限制为
            bullet_sub_id = bullet_class.sub_id # 细致id 1~10的数字
            player_weapon_bullet_locs = [bullet_type,bullet_subtype,bullet_sub_id]
            
            player_weapon_list[player_weapon.slot_id - 1] = [player_weapon_is_active, 
                                                                player_weapon_type, 
                                                                player_weapon_subtype,
                                                                player_weapon_sub_id,
                                                                player_weapon_current_bullet,
                                                                player_weapon_rest_bullet,
                                                                player_weapon_capacity,
                                                                player_weapon_bullet_percent,
                                                                player_weapon_remain_reloading,]+ \
                                                                player_weapon_attachments_locs + \
                                                                player_weapon_bullet_locs
            
        all_player_weapon = torch.as_tensor(player_weapon_list) # .reshape(-1, 27)  # in case sorted_door_list is empty
        # player_weapon_padding_num = self.max_player_weapon_num - len(all_player_weapon)
        # player_weapon_num = len(all_player_weapon)
        # all_player_weapon = torch.nn.functional.pad(all_player_weapon, (0, 0, 0, player_weapon_padding_num), 'constant', 0)
        player_weapon_info = {
            'is_active': all_player_weapon[:, 0].long(), # 是否掏出武器
            'maintype': all_player_weapon[:, 1].long(), # 大类  1~10可限制为 bianry 5
            'subtype': all_player_weapon[:, 2].long(), # 细分小类 1~10可限制为   bianry 5
            'sub_id': all_player_weapon[:,3].long(), # 细致id 1~10的数字  bianry 5
            'bullet_current': ((all_player_weapon[:, 4].clamp(max=self.bullet_norm))/ self.bullet_norm).float(), # 子弹数目 使用100进行归一化
            'bullet_rest': ((all_player_weapon[:, 5].clamp(max=self.bullet_norm))/ self.bullet_norm).float(), # 子弹数目 使用100进行归一化
            'capacity': ((all_player_weapon[:, 6].clamp(max=self.bullet_norm))/ self.bullet_norm).float(), # 弹夹容量 使用100进行归一化
            'bullet_percent': all_player_weapon[:, 7].float(), # 子弹占弹夹容量百分比
            # 'bullet2backpack_percent': all_player_weapon[:, 8].long(), # 子弹占弹备弹百分比
            'remain_reloading': (all_player_weapon[:, 8]/self.reloading_time_norm).float(), # 剩余填充时间
            # 'can_use_attachments': all_player_weapon[:,8:25].long(), # 该枪可使用哪些配件 可移除
            # 'used_attachments': all_s_player_weapon[:,25:41].long(), # 该枪已装配上哪些配件 可移除
            
            'Muzzle_main_type': all_player_weapon[:,9].long(), # 配件 大类 bianry 5
            'Muzzle_subtype': all_player_weapon[:,10].long(), # 配件 细分小类
            'Muzzle_sub_id': all_player_weapon[:,11].long(), # 配件 细致id
            'grip_main_type': all_player_weapon[:,12].long(),
            'grip_subtypee': all_player_weapon[:,13].long(),
            'grip_sub_id': all_player_weapon[:,14].long(),
            'butt_main_type': all_player_weapon[:,15].long(),
            'butt_subtype': all_player_weapon[:,16].long(),
            'butt_sub_id': all_player_weapon[:,17].long(),
            'clip_main_type': all_player_weapon[:,18].long(),
            'clip_subtype': all_player_weapon[:,19].long(),
            'clip_sub_id': all_player_weapon[:,20].long(),
            'sight_main_type': all_player_weapon[:,21].long(),
            'sight_subtype': all_player_weapon[:,22].long(),
            'sight_sub_id': all_player_weapon[:,23].long(),
            'bullet_main_type': all_player_weapon[:,24].long(), # 可使用子弹的category bianry 5
            'bullet_subtype': all_player_weapon[:,25].long(),
            'bullet_sub_id': all_player_weapon[:,26].long(),
            'player_weapon_num': torch.tensor(player_weapon_num, dtype=torch.long), # 武器数量 使用transformer进行encoder时可以使用
        }
            
        self.reward_info["player_weapon_num"] = player_weapon_num

        return player_weapon_info,item_id2item_category,\
                activate_slot_id,activate_slot_bullet,\
                activate_slot_capacity,max_weapon_remain_reloading,weapon_mask

    def transform_monster_info(self, state,own_position):

        # ===============
        # monster @wj
        # ===============
        
        ### 预先将所有野怪的信息进行存储
        
        monsters = state.monsters
        monsters_list = []
        
        for monster in monsters:
            mon_position = monster.position
            mon_pos_x = mon_position.x
            mon_pos_y = mon_position.y
            mon_pos_z = mon_position.z
            mon2own_distance = self.computer_distance(mon_pos_x,mon_pos_y,mon_pos_z,own_position.x,own_position.y,own_position.z)
            if  mon2own_distance > self.monsters_visble_distance:
                continue
            if monster.monster_type in self.MONSTER.keys():
                mon_type = self.MONSTER[monster.monster_type]
            else:
                mon_type = len(self.MONSTER) + 1
            mon_max_hp = monster.max_hp
            mon_cur_hp = monster.cur_hp
            
            mon_relative_me_pos_x = mon_position.x - own_position.x
            mon_relative_me_pos_y = mon_position.y - own_position.y
            mon_relative_me_pos_z = mon_position.z - own_position.z
            
            mon_rotation_x = monster.rotation.x  
            mon_rotation_x_sin = math.sin(mon_rotation_x*math.pi/180)
            mon_rotation_x_cos = math.cos(mon_rotation_x*math.pi/180)
            mon_rotation_y = monster.rotation.y  
            mon_rotation_y_sin = math.sin(mon_rotation_y*math.pi/180)
            mon_rotation_y_cos = math.cos(mon_rotation_y*math.pi/180)
            mon_rotation_z = monster.rotation.z 
            mon_rotation_z_sin = math.sin(mon_rotation_z*math.pi/180)
            mon_rotation_z_cos = math.cos(mon_rotation_z*math.pi/180)
            
            
            mon_size_x = monster.size.x  
            mon_size_y = monster.size.y  
            mon_size_z = monster.size.z
            mon_target_id = monster.target_id
            ## TO DO QUDIAO
            if mon_target_id == self.id:
                mon_target_player = 0 # 自己
            elif mon_target_id in self.teammate_dict[self.id]:
                mon_target_player = 1 # 队友
            else:
                mon_target_player = 2 # 谁也没打
            # else:
            #     mon_target_player = 3 # 敌人
            
            monsters_list.append(
                [
                    mon2own_distance,
                    mon_type, # 5位二进制编码
                    mon_max_hp,
                    mon_cur_hp,
                    mon_pos_x,
                    mon_pos_y,
                    mon_pos_z,
                    mon_relative_me_pos_x,
                    mon_relative_me_pos_y,
                    mon_relative_me_pos_z,
                    mon_rotation_x,
                    mon_rotation_x_sin,
                    mon_rotation_x_cos,
                    mon_rotation_y,
                    mon_rotation_y_sin,
                    mon_rotation_y_cos,
                    mon_rotation_z,
                    mon_rotation_z_sin,
                    mon_rotation_z_cos,
                    mon_size_x,
                    mon_size_y,
                    mon_size_z,
                    mon_target_player # 4位one hot
                ]
            )
        
        sorted_monsters_list = sorted(monsters_list)
        all_monster = torch.as_tensor(sorted_monsters_list).reshape(-1, 23) 
        monster_padding_num = self.max_monster_num - len(all_monster)
        monster_num = len(all_monster)
        all_monster = torch.nn.functional.pad(all_monster, (0, 0, 0, monster_padding_num), 'constant', 0)
            
        
        
        monster_info = {
            
                    "mon2own_distance":all_monster[:,0].float() / self.monsters_visble_distance, 
                    "mon_type":all_monster[:,1].long(), # 5位二进制编码
                    "mon_max_hp" :all_monster[:,2].float()/ self.monster_hp_norm,
                    "mon_cur_hp":all_monster[:,3].float()/ self.monster_hp_norm,
                    "mon_cur_hp_percent":all_monster[:,3] / ((all_monster[:,2]).float() + 1e-9),
                    "mon_pos_x":all_monster[:,4].float() / self.mapsize[0],
                    "mon_pos_y":all_monster[:,5].float()/ self.mapsize[1],
                    "mon_pos_z":all_monster[:,6].float()/ self.mapsize[2],
                    "mon_relative_me_pos_x":all_monster[:,7].float() / (all_monster[:,0] + 1e-9),
                    "mon_relative_me_pos_y":all_monster[:,8].float() / (all_monster[:,0] + 1e-9),
                    "mon_relative_me_pos_z":all_monster[:,9].float() / (all_monster[:,0] + 1e-9),
                    "mon_rotation_x":all_monster[:,10].float() / self.rotation_norm, 
                    "mon_rotation_x_sin":all_monster[:,11].float() ,
                    "mon_rotation_x_cos":all_monster[:,12].float() ,
                    "mon_rotation_y":all_monster[:,13].float() / self.rotation_norm ,
                    "mon_rotation_y_sin":all_monster[:,14].float(),
                    "mon_rotation_y_cos":all_monster[:,15].float(),
                    "mon_rotation_z":all_monster[:,16].float() / self.rotation_norm,
                    "mon_rotation_z_sin":all_monster[:,17].float(),
                    "mon_rotation_z_cos":all_monster[:,18].float(),
                    "mon_size_x":all_monster[:,19].float() / self.size_xy_norm,
                    "mon_size_y":all_monster[:,20].float() / self.size_xy_norm,
                    "mon_size_z":all_monster[:,21].float()/ self.size_z_norm,
                    "mon_target_player":all_monster[:,22].long(), # 3位数的one hot 编码
                    "monster_num":torch.tensor(monster_num, dtype=torch.long)
        }
        return monster_info


    def transform_door_info(self,info,own_position):
            
        # ===============
        # doors  @wj
        # ===============
        
        doors = info['doors']
        door_list = []
        
        # 预先将上千道门的位置信息进行储存，后续使用矩阵运算筛选出可见距离内的门！
        if self.doors_pos_matrix is None or len(self.doors_pos_matrix)==0:
            self.doors_pos_matrix = []
            self.doors_categorys = []
            for door_id, door in doors.items():
                self.doors_categorys.append(door_id)
                self.doors_pos_matrix.append([door.position.x,door.position.y,door.position.z])
            self.doors_pos_matrix = torch.as_tensor(self.doors_pos_matrix)
            self.doors_categorys = torch.as_tensor(self.doors_categorys)
        # 找出范围内的door
        visble_doors = self.find_nearest_doors([own_position.x,own_position.y,own_position.z])
        
        nearest_door_dist = np.inf
        yaw_door_position = [[] for _ in range(len(self.heat_map_deltas["yaw_delta_10"]))]
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
            door_pitch, door_yaw = get_picth_yaw(door_rel_x, door_rel_y, door_rel_z)
            # 1：开，0：关闭
            door_state = door.state
            # 距离门的距离
            distance = math.sqrt(door_rel_x ** 2 + door_rel_y ** 2 +
                                 door_rel_z ** 2)
            # 门的类型
            door_type = door.door_type
            nearest_door_dist = min(nearest_door_dist,distance)

            for delta,pool_map_size in self.all_pool_map_size.items():
                if delta in self.heat_map_deltas["yaw_delta_10"]:
                    if abs(door_rel_x) <= pool_map_size[0]*self.heat_map_scale \
                        and abs(door_rel_y) <= pool_map_size[1]*self.heat_map_scale\
                        and abs(door_rel_z+35) <= 200:
                        pos = [int(door_pos.x//self.heat_map_scale),
                                int(door_pos.y//self.heat_map_scale),door_state]
                        
                        yaw_door_position[self.heat_map_deltas["yaw_delta_10"].index(delta)].append(pos)
            door_list.append([
                distance,
                door_pos.x,
                door_pos.y,
                door_pos.z,
                door_rel_x,
                door_rel_y,
                door_rel_z,
                door_pitch,
                door_yaw,
                door_state,
                door_type,
            ])
        sorted_door_list = sorted(door_list)
        all_door = torch.as_tensor(sorted_door_list).reshape(-1, 11) 
        door_padding_num = self.max_door_num - len(all_door)
        door_num = len(all_door)
        all_door = torch.nn.functional.pad(all_door, (0, 0, 0, door_padding_num), 'constant', 0)
        
        door_info = {
            'distance': (all_door[:, 0] / self.doors_visble_distance).float(), # 除以door的范围可见距离
            'x': (all_door[:, 1] / self.mapsize[0]).float(),    # 除以统一的距离归一化值
            'y': (all_door[:, 2] / self.mapsize[1]).float(),    # 除以统一的距离归一化值
            'z': (all_door[:, 3] / self.mapsize[2]).float(),    # 除以统一的距离归一化值
            're_x': (all_door[:, 4] / (all_door[:, 0] + 1e-7)).float(), # 除以door的范围可见距离
            're_y': (all_door[:, 5] / (all_door[:, 0] + 1e-7)).float(), # 除以door的范围可见距离
            're_z': (all_door[:, 6] / (all_door[:, 0] + 1e-7)).float(),  # 除以door的范围可见距离
            
            'pitch': (all_door[:, 7] / self.rotation_norm).float() ,   # 除以180度
            'yaw': (all_door[:, 8] / self.rotation_norm).float(),    # 除以180度
            'door_state': (all_door[:, 9]).long(), # 二值 不需要处理
            'door_type': (all_door[:, 10]).long(), # 4位二进制编码
            'door_num': torch.tensor(door_num, dtype=torch.long), # 使用transformer进行encoder时可以使用
            
        }
        return door_info,nearest_door_dist,yaw_door_position
    
    def transform_only_v_info(self,safety_area,state_info,info,own_player_state,own_position,state,search_target_player_id=None):
        #### enemy info for v network
        v_enemy_list = []
        safety_area_pos_x = safety_area.center.x 
        safety_area_pos_y = safety_area.center.y  
        safety_area_radius = safety_area.radius  
        safety_area_next_pos_x = safety_area.next_center.x  
        safety_area_next_pos_y = safety_area.next_center.y  
        safety_area_next_radius = safety_area.next_radius  
        safety_area_rest_time = safety_area.total_time - safety_area.time  # rest_time
        since_last_see_time = 0
        cur_time = state.timestamp
        
        # heatmap
        top2bottom_enemy_position = [[] for _ in range(len(self.heat_map_deltas["top2bottom_10"]))]
        pitch_enemy_position = [[] for _ in range(len(self.heat_map_deltas["pitch_delta_10"]))]
        yaw_enemy_position = [[] for _ in range(len(self.heat_map_deltas["yaw_delta_10"]))]
        out_area_player_id = None
        for player_id, player_item in state_info.items():
            if player_id == self.id  or \
               info['player_state'][player_id].state.team_id == own_player_state.state.team_id or \
               not info['alive_players'][player_id]:               
                continue    
                 
            enemy_character = all_characters[player_item.state.actor_id].id - 1100
            enemy_pos = player_item.state.position
            enemy_pos_x = player_item.state.position.x
            enemy_pos_y = player_item.state.position.y
            enemy_pos_z = player_item.state.position.z
            enemy_relative_me_pos_x = enemy_pos.x - own_position.x
            enemy_relative_me_pos_y = enemy_pos.y - own_position.y
            enemy_relative_me_pos_z = enemy_pos.z - own_position.z
            
            
            
            # 看得见才添加
            if player_id  in own_player_state.visble_player_ids or player_id==search_target_player_id:
                
                self.heatmap_enemy_position[player_id] = [enemy_pos_x,enemy_pos_y,cur_time]
                
            
            enemy_distance = math.sqrt(enemy_relative_me_pos_x ** 2 + enemy_relative_me_pos_y ** 2 + enemy_relative_me_pos_z ** 2)
            if enemy_distance > self.v_inspect_enemy_distance and  player_id not in own_player_state.visble_player_ids:
                if out_area_player_id is None:
                    out_area_player_id = [player_id, enemy_distance]
                else:
                    if enemy_distance < out_area_player_id[1]:
                        out_area_player_id = [player_id, enemy_distance]
                continue
            enemy_rotation_x = player_item.state.rotation.x  
            enemy_rotation_x_sin = math.sin(enemy_rotation_x*math.pi/180)
            enemy_rotation_x_cos = math.cos(enemy_rotation_x*math.pi/180)
            enemy_rotation_y = player_item.state.rotation.y  
            enemy_rotation_y_sin = math.sin(enemy_rotation_y*math.pi/180)
            enemy_rotation_y_cos = math.cos(enemy_rotation_y*math.pi/180)
            enemy_rotation_z = player_item.state.rotation.z 
            enemy_rotation_z_sin = math.sin(enemy_rotation_z*math.pi/180)
            enemy_rotation_z_cos = math.cos(enemy_rotation_z*math.pi/180)
            enemy_size_x = player_item.state.size.x  
            enemy_size_y = player_item.state.size.y  
            enemy_size_z = player_item.state.size.z
            enemy_speed_x = player_item.state.speed.x  
            enemy_speed_y = player_item.state.speed.y  
            enemy_speed_z = player_item.state.speed.z
            enemy_scalar_speed = min(math.sqrt( enemy_speed_x**2 + enemy_speed_y**2 + enemy_speed_z**2 ), 1000)
            enemy_hp = player_item.state.hp 
            enemy_neardeath_breath = player_item.state.neardeath_breath 
            enemy_oxygen = player_item.state.oxygen  
            enemy_peek = player_item.state.peek_type  
            enemy_alive = player_item.state.alive_state  
            enemy_bodystate = player_item.state.body_state  


            enemy_team_id = player_item.state.team_id

            try:
                enemy_see_me = self.id in state_info[player_id].visble_player_ids
            except:
                enemy_see_me = False
            enemy_relative_blue_safetyarea_x,enemy_relative_blue_safetyarea_y, _ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_pos_x, safety_area_pos_y,0 ) 
            enemy_relative_white_safetyarea_x,enemy_relative_white_safetyarea_y,_ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_next_pos_x, safety_area_next_pos_y, 0) 
            enemy_distance_blue_safetyarea = math.sqrt(enemy_relative_blue_safetyarea_x**2+enemy_relative_blue_safetyarea_y**2)
            enemy_distance_white_safetyarea = math.sqrt(enemy_relative_white_safetyarea_x**2+enemy_relative_white_safetyarea_y**2)
            enemy_in_blue_safetyarea = enemy_distance_blue_safetyarea < safety_area_radius                                        
            enemy_in_white_safetyarea = enemy_distance_white_safetyarea < safety_area_next_radius
            whether_enemy_run_in_blue_circle_time = enemy_distance_blue_safetyarea / (self.expect_speed)-safety_area_rest_time
            whether_enemy_run_in_blue_circle = enemy_distance_blue_safetyarea / (self.expect_speed)
            whether_enemy_run_in_white_circle = enemy_distance_white_safetyarea / (self.expect_speed)
            
            hold_gun = 1 if len(player_item.weapon.player_weapon) > 0 else 0
            
            enemy_buff_list = [0] * 3
            for buff_idx,i in enumerate(player_item.state.buff):
                if buff_idx >= 3:
                    break
                enemy_buff_list[buff_idx] = i+1

            enemy_info = [enemy_distance,  ##0
                               enemy_team_id,
                               enemy_pos_x,
                               enemy_pos_y,
                               enemy_pos_z,
                               enemy_rotation_x,
                               enemy_rotation_y,
                               enemy_rotation_z,
                               enemy_rotation_x_sin,
                               enemy_rotation_y_sin,
                               enemy_rotation_z_sin,   ##10
                               enemy_rotation_x_cos,
                               enemy_rotation_y_cos,
                               enemy_rotation_z_cos,
                               enemy_size_x,
                               enemy_size_y,
                               enemy_size_z,
                               enemy_speed_x,
                               enemy_speed_y,
                               enemy_speed_z,
                               enemy_scalar_speed,    ##20
                               enemy_hp,
                               enemy_neardeath_breath,
                               enemy_oxygen,
                               enemy_peek,
                               enemy_alive,
                               enemy_bodystate,
                               enemy_relative_me_pos_x,
                               enemy_relative_me_pos_y,
                               enemy_relative_me_pos_z,                               
                               enemy_character,        ##30               
                               enemy_see_me,                               
                               enemy_relative_blue_safetyarea_x,
                               enemy_relative_blue_safetyarea_y,
                               enemy_relative_white_safetyarea_x,
                               enemy_relative_white_safetyarea_y,
                               enemy_distance_blue_safetyarea,
                               enemy_distance_white_safetyarea,
                               enemy_in_blue_safetyarea,
                               enemy_in_white_safetyarea,
                               whether_enemy_run_in_blue_circle_time,   ##40
                               whether_enemy_run_in_blue_circle,
                               whether_enemy_run_in_white_circle,
                               hold_gun,
                               since_last_see_time,
                               ] + enemy_buff_list

            v_enemy_list.append(enemy_info)
        for p_id,pos_info in self.heatmap_enemy_position.items():
            enemy_relative_me_pos_x = pos_info[0] - own_position.x
            enemy_relative_me_pos_y = pos_info[1] - own_position.y
            for delta,pool_map_size in self.all_pool_map_size.items():
                if abs(enemy_relative_me_pos_x) <= pool_map_size[0]*self.heat_map_scale \
                    and abs(enemy_relative_me_pos_y) <= pool_map_size[1]*self.heat_map_scale\
                    and cur_time - pos_info[2] <= self.heatmap_enemy_time_delta * 1000:
                    pos = [int(pos_info[0]//self.heat_map_scale),
                            int(pos_info[1]//self.heat_map_scale),(cur_time - pos_info[2])/1000]
                    if delta in self.heat_map_deltas["top2bottom_10"]:
                        top2bottom_enemy_position[self.heat_map_deltas["top2bottom_10"].index(delta)].append(pos)
                    if delta in self.heat_map_deltas["pitch_delta_10"]:
                        pitch_enemy_position[self.heat_map_deltas["pitch_delta_10"].index(delta)].append(pos)
                    if delta in self.heat_map_deltas["yaw_delta_10"]:
                        yaw_enemy_position[self.heat_map_deltas["yaw_delta_10"].index(delta)].append(pos)

        if len(v_enemy_list) == 0:
            if out_area_player_id is not None:
                player_id = out_area_player_id[0]
                player_item = state_info[player_id]

                enemy_character = all_characters[player_item.state.actor_id].id - 1100
                enemy_pos = player_item.state.position
                enemy_pos_x = player_item.state.position.x
                enemy_pos_y = player_item.state.position.y
                enemy_pos_z = player_item.state.position.z
                enemy_relative_me_pos_x = enemy_pos.x - own_position.x
                enemy_relative_me_pos_y = enemy_pos.y - own_position.y
                enemy_relative_me_pos_z = enemy_pos.z - own_position.z
                enemy_rotation_x = player_item.state.rotation.x  
                enemy_rotation_x_sin = math.sin(enemy_rotation_x*math.pi/180)
                enemy_rotation_x_cos = math.cos(enemy_rotation_x*math.pi/180)
                enemy_rotation_y = player_item.state.rotation.y  
                enemy_rotation_y_sin = math.sin(enemy_rotation_y*math.pi/180)
                enemy_rotation_y_cos = math.cos(enemy_rotation_y*math.pi/180)
                enemy_rotation_z = player_item.state.rotation.z 
                enemy_rotation_z_sin = math.sin(enemy_rotation_z*math.pi/180)
                enemy_rotation_z_cos = math.cos(enemy_rotation_z*math.pi/180)
                enemy_size_x = player_item.state.size.x  
                enemy_size_y = player_item.state.size.y  
                enemy_size_z = player_item.state.size.z
                enemy_speed_x = player_item.state.speed.x  
                enemy_speed_y = player_item.state.speed.y  
                enemy_speed_z = player_item.state.speed.z
                enemy_scalar_speed = min(math.sqrt( enemy_speed_x**2 + enemy_speed_y**2 + enemy_speed_z**2 ), 1000)
                enemy_hp = player_item.state.hp 
                enemy_neardeath_breath = player_item.state.neardeath_breath 
                enemy_oxygen = player_item.state.oxygen  
                enemy_peek = player_item.state.peek_type  
                enemy_alive = player_item.state.alive_state  
                enemy_bodystate = player_item.state.body_state  

                enemy_team_id = player_item.state.team_id

                try:
                    enemy_see_me = self.id in state_info[player_id].visble_player_ids
                except:
                    enemy_see_me = False
                enemy_relative_blue_safetyarea_x,enemy_relative_blue_safetyarea_y, _ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_pos_x, safety_area_pos_y,0 ) 
                enemy_relative_white_safetyarea_x,enemy_relative_white_safetyarea_y,_ = self.vec_of_2position(enemy_pos_x, enemy_pos_y, 0, safety_area_next_pos_x, safety_area_next_pos_y, 0) 
                enemy_distance_blue_safetyarea = math.sqrt(enemy_relative_blue_safetyarea_x**2+enemy_relative_blue_safetyarea_y**2)
                enemy_distance_white_safetyarea = math.sqrt(enemy_relative_white_safetyarea_x**2+enemy_relative_white_safetyarea_y**2)
                enemy_in_blue_safetyarea = enemy_distance_blue_safetyarea < safety_area_radius                                        
                enemy_in_white_safetyarea = enemy_distance_white_safetyarea < safety_area_next_radius
                whether_enemy_run_in_blue_circle_time = enemy_distance_blue_safetyarea / (self.expect_speed)-safety_area_rest_time
                whether_enemy_run_in_blue_circle = enemy_distance_blue_safetyarea / (self.expect_speed)
                whether_enemy_run_in_white_circle = enemy_distance_white_safetyarea / (self.expect_speed)
                
                hold_gun = 1 if len(player_item.weapon.player_weapon) > 0 else 0
                
                enemy_buff_list = [0] * 3
                for buff_idx,i in enumerate(player_item.state.buff):
                    if buff_idx >= 3:
                        break
                    enemy_buff_list[buff_idx] = i+1

                enemy_info = [enemy_distance,  ##0
                                enemy_team_id,
                                enemy_pos_x,
                                enemy_pos_y,
                                enemy_pos_z,
                                enemy_rotation_x,
                                enemy_rotation_y,
                                enemy_rotation_z,
                                enemy_rotation_x_sin,
                                enemy_rotation_y_sin,
                                enemy_rotation_z_sin,   ##10
                                enemy_rotation_x_cos,
                                enemy_rotation_y_cos,
                                enemy_rotation_z_cos,
                                enemy_size_x,
                                enemy_size_y,
                                enemy_size_z,
                                enemy_speed_x,
                                enemy_speed_y,
                                enemy_speed_z,
                                enemy_scalar_speed,    ##20
                                enemy_hp,
                                enemy_neardeath_breath,
                                enemy_oxygen,
                                enemy_peek,
                                enemy_alive,
                                enemy_bodystate,
                                enemy_relative_me_pos_x,
                                enemy_relative_me_pos_y,
                                enemy_relative_me_pos_z,                               
                                enemy_character,        ##30               
                                enemy_see_me,                               
                                enemy_relative_blue_safetyarea_x,
                                enemy_relative_blue_safetyarea_y,
                                enemy_relative_white_safetyarea_x,
                                enemy_relative_white_safetyarea_y,
                                enemy_distance_blue_safetyarea,
                                enemy_distance_white_safetyarea,
                                enemy_in_blue_safetyarea,
                                enemy_in_white_safetyarea,
                                whether_enemy_run_in_blue_circle_time,   ##40
                                whether_enemy_run_in_blue_circle,
                                whether_enemy_run_in_white_circle,
                                hold_gun,
                                since_last_see_time,
                                ] + enemy_buff_list

                v_enemy_list.append(enemy_info)


        v_sorted_enemy_list = sorted(v_enemy_list)
        v_all_enemy_item = torch.as_tensor(v_sorted_enemy_list).reshape(-1, 48)  # in case sorted_supply_item_list is empty
        v_enemy_item_num = len(v_all_enemy_item)
        v_enemy_item_padding_num = self.max_enemy_num - len(v_all_enemy_item)
        
        v_all_enemy_item = torch.nn.functional.pad(v_all_enemy_item, (0, 0, 0, v_enemy_item_padding_num), 'constant', 0)
        # all_enemy_item = torch.as_tensor(enemy_list).reshape(-1, 5)
        v_enemy_item_info = {
            'distance':v_all_enemy_item[:,0].float() / self.mapsize[0],              
            'team_id':v_all_enemy_item[:,1].long(),  ##TODO, binary 7, learner
            'pos_x':v_all_enemy_item[:,2].float() / self.mapsize[0],
            'pos_y':v_all_enemy_item[:,3].float() / self.mapsize[1],
            'pos_z':v_all_enemy_item[:,4].float() / self.mapsize[2],
            'rotation_x':v_all_enemy_item[:,5].float() / self.rotation_norm,
            'rotation_y':v_all_enemy_item[:,6].float() / self.rotation_norm,
            'rotation_z':v_all_enemy_item[:,7].float() / self.rotation_norm,
            'rotation_sin_x':v_all_enemy_item[:,8].float(),
            'rotation_sin_y':v_all_enemy_item[:,9].float(),
            'rotation_sin_z':v_all_enemy_item[:,10].float(),
            'rotation_cos_x':v_all_enemy_item[:,11].float(),
            'rotation_cos_y':v_all_enemy_item[:,12].float(),
            'rotation_cos_z':v_all_enemy_item[:,13].float(),
            'size_x':v_all_enemy_item[:,14].float() / self.size_xy_norm,
            'size_y':v_all_enemy_item[:,15].float() / self.size_xy_norm,
            'size_z':v_all_enemy_item[:,16].float() / self.size_z_norm,
            'speed_x':v_all_enemy_item[:,17].float() / self.speed_norm,
            'speed_y':v_all_enemy_item[:,18].float() / self.speed_norm,
            'speed_z':v_all_enemy_item[:,19].float() / self.speed_norm,
            'scalar_speed':v_all_enemy_item[:,20].float() / self.speed_norm,
            'hp':v_all_enemy_item[:,21].float() / self.hp_norm,
            'neardeath_breath':v_all_enemy_item[:,22].float() / self.hp_norm,                                    
            'oxygen':v_all_enemy_item[:,23].float() / self.oxygen_norm,
            'peek':v_all_enemy_item[:,24].long(),  ##onehot 3,
            'alive':v_all_enemy_item[:,25].long(),   ##onehot 3
            'bodystate':v_all_enemy_item[:,26].long(),  ##onehot 8
            'relative_pos_x':v_all_enemy_item[:,27].float() / (v_all_enemy_item[:,0] +1e-9),
            'relative_pos_y':v_all_enemy_item[:,28].float() / (v_all_enemy_item[:,0] +1e-9),  
            'relative_pos_z':v_all_enemy_item[:,29].float() / (v_all_enemy_item[:,0] +1e-9), 
            'character':v_all_enemy_item[:,30].long(),       ### binary 5                 
            'enemy_see_me':v_all_enemy_item[:,31].long(),  ###TODO
            'enemy_relative_blue_safetyarea_x':v_all_enemy_item[:,32].float()/ self.mapsize[0],
            'enemy_relative_blue_safetyarea_y':v_all_enemy_item[:,33].float()/ self.mapsize[1],  ##TODO
            'enemy_relative_white_safetyarea_x':v_all_enemy_item[:,34].float()/ self.mapsize[0],
            'enemy_relative_white_safetyarea_y':v_all_enemy_item[:,35].float()/ self.mapsize[1],
            'enemy_distance_blue_safetyarea':v_all_enemy_item[:,36].float()/ self.mapsize[0],
            'enemy_distance_blue_safetyarea_relative':v_all_enemy_item[:,36].float()/(safety_area_radius + 1),
            'enemy_distance_white_safetyarea':v_all_enemy_item[:,37].float()/ self.mapsize[0],
            'enemy_distance_white_safetyarea_relative':v_all_enemy_item[:,37].float()/(safety_area_next_radius + 1),
            'enemy_in_blue_safetyarea':v_all_enemy_item[:,38].long(),
            'enemy_in_white_safetyarea':v_all_enemy_item[:,39].long(),
            'whether_enemy_run_in_blue_circle_time':(v_all_enemy_item[:,40].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'whether_enemy_run_in_blue_circle':(v_all_enemy_item[:,41].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),
            'whether_enemy_run_in_white_circle':(v_all_enemy_item[:,42].float()/self.MAX_GAME_TIME).clamp(min=-5,max=5),
            "hold_gun": v_all_enemy_item[:, 43].long() ,                
            'since_last_see_time':v_all_enemy_item[:,44].float()/self.MAX_GAME_TIME,  
            'enemy_buff_1': v_all_enemy_item[:, 45].long() ,
            'enemy_buff_2': v_all_enemy_item[:, 46].long() ,
            'enemy_buff_3': v_all_enemy_item[:, 47].long() ,
            'enemy_item_num': torch.tensor(v_enemy_item_num, dtype=torch.long),
        }
        
        return v_enemy_item_info,top2bottom_enemy_position,pitch_enemy_position,yaw_enemy_position
    
    def transform_history_position_heatmap(self,own_player_state,top2bottom_teammate_position,
                                            pitch_teammate_position,yaw_teammate_position,
                                            top2bottom_enemy_position,pitch_enemy_position,
                                            yaw_enemy_position,yaw_door_position):
        # ##################
        # 添加一些历史轨迹信息
        # ##################
        
        self.self_history_positions.append([own_player_state.state.position.x,own_player_state.state.position.y])
        
        history_positions = list(self.self_history_positions)
        
        history_positions_info = history_positions[::10]
        history_positions_info = (torch.tensor(history_positions_info,dtype=torch.float)/self.mapsize[0]).flatten()
        
        yaw_door_position, yaw_door_nums = self.padding_positions_with_time(yaw_door_position,self.heat_map_max_door_nums)
        
        top2bottom_teammate_position, top2bottom_teammate_nums = self.padding_positions(top2bottom_teammate_position,self.heat_map_max_teammate_nums)
        
        top2bottom_enemy_position, top2bottom_enemy_nums = self.padding_positions_with_time(top2bottom_enemy_position,self.heat_map_max_enemy_nums)
        
        pitch_teammate_position, pitch_teammate_nums = self.padding_positions(pitch_teammate_position,self.heat_map_max_teammate_nums)
        
        pitch_enemy_position, pitch_enemy_nums = self.padding_positions_with_time(pitch_enemy_position,self.heat_map_max_enemy_nums)
        
        yaw_teammate_position, yaw_teammate_nums = self.padding_positions(yaw_teammate_position,self.heat_map_max_teammate_nums)
        
        yaw_enemy_position, yaw_enemy_nums = self.padding_positions_with_time(yaw_enemy_position,self.heat_map_max_enemy_nums)
        
        position_info = {
            "position_x":torch.tensor(int(own_player_state.state.position.x // self.heat_map_scale), dtype=torch.long),
            "position_y":torch.tensor(int(own_player_state.state.position.y // self.heat_map_scale), dtype=torch.long),
            "position_z":torch.tensor(int(own_player_state.state.position.z), dtype=torch.long),
            "yaw_door_position": yaw_door_position.long(),
            "top2bottom_teammate_position":top2bottom_teammate_position.long(),
            "top2bottom_enemy_position":top2bottom_enemy_position.long() ,
            
            "pitch_teammate_position":pitch_teammate_position.long(),
            "pitch_enemy_position": pitch_enemy_position.long(),
            "yaw_teammate_position":yaw_teammate_position.long(),
            "yaw_enemy_position": yaw_enemy_position.long(),
            
            "top2bottom_teammate_nums":top2bottom_teammate_nums.long(),
            "top2bottom_enemy_nums":top2bottom_enemy_nums.long(),
            "pitch_teammate_nums":pitch_teammate_nums.long(),
            "pitch_enemy_nums":pitch_enemy_nums.long(),
            "yaw_teammate_nums":yaw_teammate_nums.long(),
            "yaw_enemy_nums":yaw_enemy_nums.long(),
            "yaw_door_nums":yaw_door_nums.long(),
            "history_position": (torch.tensor(history_positions )// self.heat_map_scale).long(),
            # "server_is_0913":torch.tensor(self.cfg.game.server_init.server_mode == "0913").long(),
            "server_is_0913":torch.tensor(1).long(),
        }
        # a = time.time()
        heatmap_info = self.heatmap_tool.generate_heat_map(position_info,1)
        
        # print("generate_heat_map:",time.time()-a)
        return position_info, heatmap_info, history_positions_info
    
    
    def transform_event_info(self,state, own_player_state,own_position,state_info):
        damage_source = own_player_state.damage.damage_source
        cur_time = state.timestamp


        for damage in damage_source:
  
            if damage.damage_source_id >= 6000 and damage.damage_source_id < (6000+self.team_num*self.player_num_per_team):
                source_position = state_info[damage.damage_source_id].state.position

                damage2own_distance = self.computer_distance(source_position.x,source_position.y,source_position.z,own_position.x,own_position.y,own_position.z)

                damage_relative_me_pos_x = source_position.x - own_position.x
                damage_relative_me_pos_y = source_position.y - own_position.y
                damage_relative_me_pos_z = source_position.z - own_position.z
                self.history_event_info.append( [1,damage.type,source_position.x, # 1代表伤害
                                            source_position.y,source_position.z,
                                            damage.damage,cur_time] + \
                                            [damage_relative_me_pos_x/(damage2own_distance+1e-7),
                                             damage_relative_me_pos_y/(damage2own_distance+1e-7),
                                             damage_relative_me_pos_z/(damage2own_distance+1e-7),0]) # 预留的四个位置
        


        # sounds = own_player_state.sound.heard_sound
        # sound_list = []
        # for sound in sounds:
        #     sound_list.append([2,sound.type,sound.location.x, # 2代表声音
        #                        sound.location.y,
        #                        sound.location.z,
        #                        0,
        #                        cur_time] +[0,0,0,0] ) 
        
        # for v in sound_list:
        #     self.history_event_info.append(v)
        
        history_event_info = list(copy.deepcopy(self.history_event_info))
        for i in range(len(history_event_info)):
            history_event_info[i][6] = cur_time - history_event_info[i][6]
        history_event_info = torch.as_tensor(history_event_info).reshape(-1, 11) 
        event_padding_num = self.MAX_Event_Num - len(history_event_info)
        event_num = len(history_event_info)
        history_event_info = torch.nn.functional.pad(history_event_info, (0, 0, 0, event_padding_num), 'constant', 0)
        
        event_info = {
            'main_type':  (history_event_info[:, 0] ).long(),  # 暂且先用三位二进制编码
            'sub_type': (history_event_info[:, 1] ).long(), # 暂且先用四位二进制编码
            'x': (history_event_info[:, 2] / self.mapsize[0]).float(),    # 除以统一的距离归一化值
            'y': (history_event_info[:, 3] / self.mapsize[1]).float(),    # 除以统一的距离归一化值
            'z': (history_event_info[:, 4] / self.mapsize[2]).float(),    # 除以统一的距离归一化值       
            'damage': (history_event_info[:, 5] / self.hp_norm).float() ,   # 
            'time_delta': (torch.clip((history_event_info[:, 6] ) / (1000*self.MAX_GAME_TIME),0,1)).float(),    # 
            'tmp_1': (history_event_info[:, 7] ),    # 
            'tmp_2': (history_event_info[:, 8]).float(),    # 
            'tmp_3': (history_event_info[:, 9]).float(),    # 
            'tmp_4': (history_event_info[:, 10]).float(),    #  # 
            'event_num': torch.tensor(event_num, dtype=torch.long), # 使用transformer进行encoder时可以使用
        }
        return event_info
    

    def transform_rotation_info(self,state_info, state,own_player_state,own_position):

        my_state = state_info[self.id].state

        pos_x = my_state.position.x
        pos_y = my_state.position.y
        pos_z = my_state.position.z
        rotation_x = my_state.rotation.x  
        rotation_y = my_state.rotation.y
        rotation_z = my_state.rotation.z  

        ## 第一次，没有历史rotation，假定当前的rotation为历史rotation
        if self.rotation_change_time == 0:
            self.rotation_change_time = self.game_start_time

        if len(self.history_rotation)==0:
            history_rotation_x = rotation_x
            history_rotation_y = rotation_y
            history_rotation_z = rotation_z
            self.history_rotation.append([rotation_x, rotation_y, rotation_z])            
        else:
            history_rotation_x = self.history_rotation[0][0]
            history_rotation_y = self.history_rotation[0][1]
            history_rotation_z = self.history_rotation[0][2]
        delta_rotation_x = rotation_x - history_rotation_x
        delta_rotation_y = rotation_y - history_rotation_y
        delta_rotation_z = rotation_z - history_rotation_z


        time = state.timestamp - self.rotation_change_time



        see_enemy =  0
        for play_id in own_player_state.visble_player_ids:
            if play_id not in self.teammate_dict[self.id] and play_id!= self.id:
                see_enemy = 1
                break

        current_delta_rotation_x = 0
        current_delta_rotation_y = 0
        current_delta_rotation_z = 0
        rotation_info_list = [
            pos_x,  #0
            pos_y,
            pos_z,
            rotation_x,
            rotation_y,
            rotation_z,
            history_rotation_x,
            history_rotation_y,
            history_rotation_z,
            delta_rotation_x,
            delta_rotation_y,
            delta_rotation_z,  #11
            current_delta_rotation_x,
            current_delta_rotation_y,
            current_delta_rotation_z,  
            time,
            see_enemy,

        ] 
        if delta_rotation_x!=0 or  delta_rotation_y!=0 or delta_rotation_z!=0:
            self.rotation_info_all_list.append(rotation_info_list) 

        if delta_rotation_x!=0 or  delta_rotation_y!=0 or delta_rotation_z!=0:
            self.rotation_change_time = state.timestamp
            self.history_rotation.append([rotation_x, rotation_y, rotation_z ])

        if len(self.rotation_info_all_list) > 0:
            v_all_rotation_item = torch.as_tensor( self.rotation_info_all_list ).reshape(-1, 17)  # in case sorted_supply_item_list is empty
            v_all_rotation_item[:, 12] = rotation_x - v_all_rotation_item[:, 6]
            v_all_rotation_item[:, 13] = rotation_y - v_all_rotation_item[:, 7]
            v_all_rotation_item[:, 14] = rotation_z - v_all_rotation_item[:, 8]
            v_rotation_item_num = len(v_all_rotation_item)
            v_rotation_item_padding_num = self.MAX_ROTATION_Num - v_rotation_item_num 
            distance = (torch.sqrt((v_all_rotation_item[:, 0] - pos_x)**2 + (v_all_rotation_item[:, 1] - pos_y)**2 + (v_all_rotation_item[:, 2]-pos_z)**2)).unsqueeze(1)
            delta_pos_x = ((v_all_rotation_item[:, 0] - pos_x).unsqueeze(1)/(distance + 1e-7))
            delta_pos_y = ((v_all_rotation_item[:, 1] - pos_y).unsqueeze(1)/(distance + 1e-7))
            delta_pos_z = ((v_all_rotation_item[:, 2] - pos_z).unsqueeze(1)/(distance + 1e-7))
            v_all_rotation_item = torch.cat([v_all_rotation_item,delta_pos_x,delta_pos_y,delta_pos_z,distance/ self.mapsize[0]],dim=1)
        else:
            v_all_rotation_item = torch.as_tensor( self.rotation_info_all_list ).reshape(-1, 21)  # in case sorted_supply_item_list is empty
            v_rotation_item_num = len(v_all_rotation_item)
            v_rotation_item_padding_num = self.MAX_ROTATION_Num - v_rotation_item_num 

        v_all_rotation_item = torch.nn.functional.pad(v_all_rotation_item, (0, 0, 0, v_rotation_item_padding_num), 'constant', 0)
        # all_enemy_item = torch.as_tensor(enemy_list).reshape(-1, 5)
        # v_enemy_item_info = {
        #     'distance':v_all_enemy_item[:,0].float() / self.mapsize[0],              

        # history_event_info = list(copy.deepcopy(self.history_event_info))
        # for i in range(len(history_event_info)):
        #     history_event_info[i][6] = time - history_event_info[i][6]
        # history_event_info = torch.as_tensor(history_event_info).reshape(-1, 11) 
        # event_padding_num = self.MAX_Event_Num - len(history_event_info)
        # rotation_num = len(history_event_info)
        # history_event_info = torch.nn.functional.pad(history_event_info, (0, 0, 0, event_padding_num), 'constant', 0)

        rotation_info = {
            'x': (v_all_rotation_item[:, 0] / self.mapsize[0]).float(),    # 除以统一的距离归一化值
            'y': (v_all_rotation_item[:, 1] / self.mapsize[1]).float(),    # 除以统一的距离归一化值
            'z': (v_all_rotation_item[:, 2] / self.mapsize[2]).float(),    # 除以统一的距离归一化值       
            'rotation_x':v_all_rotation_item[:,3].float() / self.rotation_norm,
            'rotation_y':v_all_rotation_item[:,4].float() / self.rotation_norm,
            'rotation_z':v_all_rotation_item[:,5].float() / self.rotation_norm,    
            'history_rotation_x':v_all_rotation_item[:,6].float() / self.rotation_norm,
            'history_rotation_y':v_all_rotation_item[:,7].float() / self.rotation_norm,
            'history_rotation_z':v_all_rotation_item[:,8].float() / self.rotation_norm,   
            'delta_rotation_x':v_all_rotation_item[:,9].float() / self.rotation_norm,
            'delta_rotation_y':v_all_rotation_item[:,10].float() / self.rotation_norm,
            'delta_rotation_z':v_all_rotation_item[:,11].float() / self.rotation_norm,      
            'current_delta_rotation_x':v_all_rotation_item[:,12].float() / self.rotation_norm,
            'current_delta_rotation_y':v_all_rotation_item[:,13].float() / self.rotation_norm,
            'current_delta_rotation_z':v_all_rotation_item[:,14].float() / self.rotation_norm,                               
            'time': v_all_rotation_item[:, 15].float().clamp(min=0,max=120000)/self.rotation_time_norm,  #2分钟（2x60x1000ms）来做norm 
            'see_enemy': v_all_rotation_item[:, 16].long(),    # 
            'delta_pos_x': v_all_rotation_item[:, 17].float(),    # 
            'delta_pos_y': v_all_rotation_item[:, 18].float(),    # 
            'delta_pos_z': v_all_rotation_item[:, 19].float(),    
            'distance': v_all_rotation_item[:, 20].float(), 
            'rotation_num': torch.tensor(v_rotation_item_num, dtype=torch.long), # 使用transformer进行encoder时可以使用
        }


        return rotation_info
    


    def transform_obs(self, state, objs, depth_map, last_state, last_action, info,search_target_player_id = None):
        # ================
        # 计算一些全局状态信息
        # ================

        # 统计当前场上总共有多少人
        self.all_player_nums = len(info['player_state'])


        state_info = info['player_state']
        progress_bar = state_info[self.id].progress_bar
        progress_bar_info = [
                            0, # 是否在打药包
                            0, # 剩余多久打完药包
                            0, # 是否在救队友
                            0, # 剩余多久救完队友
                            0, # 是否在被队友扶
                            0, # 剩余多久被扶起来
                            0, # 是否在换弹
                            0, # 剩余多久换好子弹
                            ]
        if progress_bar.type != 0:
            progress_bar_info[2*(progress_bar.type-1)] = 1
            progress_bar_info[2*(progress_bar.type-1) + 1] = progress_bar.remain_time
        
        # progress_bar_info[-1] = max([0] + [i.remain_reloading for i in state_info[self.id].weapon.player_weapon])
        # progress_bar_info[-2] = 1 if progress_bar_info[-1] else 0
        

        progress_bar_info[-2] = state_info[self.id].state.is_reloading
        ## todo
        if progress_bar_info[-2]:
            progress_bar_info[-1] = max([0] + [i.remain_reloading for i in state_info[self.id].weapon.player_weapon])   
        else:
            progress_bar_info[-1] = 0

        if self.game_start_time is None:
            self.game_start_time = state.timestamp
            self.not_visble_enemy_time = state.timestamp
            
        safety_area = state.safety_area

        own_player_state = state_info[self.id]
        own_position  = own_player_state.state.position

        # 统计子弹
        bullet_cnt_dict = {}
        
        # 局内id到物品id的映射
        item_id2item_category = {}
        
        # ===============
        # scalar info
        # ===============
        scalar_info,swimming_mask = self.transform_scalar_info(state_info, info, state, progress_bar_info,last_action)
        
        # ===============
        # teammate info
        # ===============
        teammate_info,teammate_alive_state,wait2saved_dist,\
            top2bottom_teammate_position,pitch_teammate_position,yaw_teammate_position = self.transform_teammate_info(info, own_player_state,state_info, safety_area)

        # ===============
        # enemy info
        # ===============
        enemy_item_info = self.transform_enemy_info(info, state, safety_area, state_info, own_player_state,search_target_player_id)        # ===============
        # visible enemy info
        # ===============
        visible_enemy_item_info = self.transform_visible_enemy_info(info, state, safety_area, state_info, own_player_state)
        # ===============
        # backpack info
        # ===============
        backpack_item_info,item_id2item_category,\
            bullet_cnt_dict, scalar_info = self.transform_backpack_info(info,item_id2item_category,bullet_cnt_dict,scalar_info)

        # ===============
        # supply info
        # ===============
        supply_item_info,scalar_info,supply_mask_distance = self.transform_supply_info(info, own_position, safety_area, state,own_player_state, scalar_info)

        # ===============
        # weapon info
        # ===============
        player_weapon_info,item_id2item_category,\
                activate_slot_id,activate_slot_bullet,\
                activate_slot_capacity,max_weapon_remain_reloading,\
                weapon_mask = self.transform_weapon_info(bullet_cnt_dict, own_player_state, item_id2item_category)

        # ===============
        # monster info
        # ===============
        monster_info = self.transform_monster_info(state,own_position)

        # ===============
        # door info
        # ===============
        door_info,nearest_door_dist,yaw_door_position = self.transform_door_info(info, own_position)
        
        # ===============
        # only_v info
        # ===============
        only_v_info,top2bottom_enemy_position,\
        pitch_enemy_position,yaw_enemy_position = self.transform_only_v_info(safety_area,state_info,info,own_player_state,own_position,state,search_target_player_id)
        
        # ===============
        # heat-map info
        # ===============
        position_info, heatmap_info, history_positions_info = self.transform_history_position_heatmap(own_player_state,top2bottom_teammate_position,
                                                                                                      pitch_teammate_position,yaw_teammate_position,
                                                                                                      top2bottom_enemy_position,pitch_enemy_position,
                                                                                                      yaw_enemy_position,yaw_door_position)
        
        # ===============
        # event_info
        # ===============
        event_info = self.transform_event_info(state, own_player_state,own_position,state_info)

        # ===============
        # rotation info
        # ===============
        rotation_info = self.transform_rotation_info(state_info, state,own_player_state,own_position)
        

        if self.cfg.actor.get("eval_mode",False) == True:
            self.visdom_heatmap = self.heatmap_tool.visdom_map

        # ======================
        # 车辆信息  vehicles  @wj
        # ======================
        # TODO: '待完成'
        vehicles_info = state.vehicles
        
        # ======================
        # depth map  @wj
        # ======================
        
        # depth, depth_geom_ids = depth_map
        # spatial_info = torch.from_numpy(depth).unsqueeze(0)
        # TODO: '将视野中的玩家id整合进入depth map'

        
        
        
        # ================
        # action mask @wj
        # ================
        mask_info = copy.deepcopy(self.action_mask)
        # 游泳的时候不能滑铲
        if swimming_mask:
            mask_info["single_head"][self.action2model_output["body_action"]["slide"]] = 1
        

        # 对开枪做mask (针对倍镜的划分下个版本加上！！！)
        self.fire_stop_mask = False
        self.fire_no_stop_mask = False
        if own_player_state.state.alive_state != 0 or \
            activate_slot_id == 0 \
            or activate_slot_bullet == 0\
            or own_player_state.state.body_state == 6:# 被打倒 没有掏出武器 or 没有子弹  or 正在游泳 
            mask_info["single_head"][self.action2model_output["items_action"]["fire"]] = 1
            mask_info["single_head"][self.action2model_output["others"]["fire_stop"]] = 1
            mask_info["single_head"][self.action2model_output["others"]["fire_stop_adjust"]] = 1

            self.fire_stop_mask = True
            self.fire_no_stop_mask = True

        

        if self.nearest_enemy_dist > self.fire_no_stop_distance: # 不符合腰射距离
            mask_info["single_head"][self.action2model_output["items_action"]["fire"]] = 1
            self.fire_no_stop_mask = True
        
        if self.nearest_enemy_dist > WEAPON_DISTANCE.get(activate_slot_id, 100000): #  射程不够
            mask_info["single_head"][self.action2model_output["others"]["fire_stop"]] = 1
            mask_info["single_head"][self.action2model_output["others"]["fire_stop_adjust"]] = 1
            self.fire_stop_mask = True
            
        if self.nearest_enemy_dist < self.fire_no_stop_distance:
            mask_info["single_head"][self.action2model_output["others"]["fire_stop_adjust"]] = 1
        


        
        # 对开门做mask
        if own_player_state.state.alive_state != 0 or nearest_door_dist > DOOR_CAN_OPEN_CLOSE_DISTANCE: # 距离不够 被打倒
            mask_info["single_head"][self.action2model_output["body_action"]["open_close_door"]] = 1
        # 对换弹做mask
        if own_player_state.state.alive_state != 0 or \
            activate_slot_id == 0 or \
            max_weapon_remain_reloading >0 \
            or activate_slot_bullet >= activate_slot_capacity: # 被打到 没有掏出武器，正在换弹，枪里子弹是满的
            mask_info["single_head"][self.action2model_output["items_action"]["reloading"]] = 1
        # 对换武器做mask
        for idx,val in enumerate(weapon_mask):
            if own_player_state.state.alive_state != 0 or \
                activate_slot_id == idx or\
                val == 0: # 不能切到当前枪 不能切到没有枪的武器槽 被打到
                mask_info["single_head"][self.action2model_output["switch_weapons"][idx]] = 1
        # 对救人做mask
        if own_player_state.state.alive_state != 0 or \
            progress_bar_info[2] or \
            teammate_alive_state != 1 or\
                wait2saved_dist > TEAMMATE_CAN_BE_SAVED_DISTANCE: # 被打到，正在救人，队友没有倒，距离不够
            mask_info["single_head"][self.action2model_output["body_action"]["rescue"]] = 1
        # 对打药包做mask
        if own_player_state.state.alive_state != 0 or \
            progress_bar_info[0] or \
            len(self.recovery_hp_items_list) == 0: # 被打到， ，正在打药, 莫得药品
            mask_info["single_head"][self.action2model_output["items_action"]["treat"]] = 1
            
        # # 对pitch做mask
        # for angle in [ -45, 0, 45]:
        #     mask_info["single_head"][self.action2model_output["pitch"][angle]] = 1

        
            
        # #　对趴下做mask,因为现在不会返回正在趴下这个state
        # mask_info["single_head"][self.action2model_output["body_action"]["ground"]] = 1
        # print(self.action2model_output["body_action"]["ground"])
        # print(mask_info["single_head"][self.action2model_output["body_action"]["ground"]])
        
        # 对捡supply做mask 如果100cm内没有supply，则mask掉
        if supply_mask_distance > 300:
            mask_info["single_head"][self.action2model_output["items_action"]["pick_up"]] = 1
        
        mask_info = {k:torch.tensor(v,dtype=torch.long) for k,v in mask_info.items()}

        
        # 添加多久没看见人的信息
        scalar_info["not_visble_enemy_time"] = torch.tensor(min((state.timestamp - self.not_visble_enemy_time) / (1000*self.MAX_GAME_TIME),1.), dtype=torch.float)

        # 添加一些场上player数量信息
        scalar_info["all_player_nums"] = torch.tensor(self.all_player_nums / 100, dtype=torch.float)
        scalar_info["alive_player_nums"] = torch.tensor(self.alive_player_nums / 100, dtype=torch.float)
        scalar_info["player_alive2all_ratio"] = torch.tensor(self.alive_player_nums / (self.all_player_nums+1e-7), dtype=torch.float)
        scalar_info["all_teammate_nums"] = torch.tensor(self.all_teammate_nums / 4, dtype=torch.float)
        scalar_info["alive_teammate_nums"] = torch.tensor(self.alive_teammate_nums / 4, dtype=torch.float)
        scalar_info["teammate_alive2all_ratio"] = torch.tensor(self.alive_teammate_nums / (self.all_teammate_nums + 1e-7), dtype=torch.float)


        features = {
            "history_positions_info":history_positions_info,
            "door_info":door_info,
            "player_weapon_info":player_weapon_info,
            "backpack_item_info":backpack_item_info,
            "supply_item_info":supply_item_info,
            "enemy_item_info":enemy_item_info,
            "visible_enemy_item_info":visible_enemy_item_info,
            "teammate_info":teammate_info,
            "scalar_info":scalar_info,
            # "spatial_info":spatial_info,
            "mask_info":mask_info,
            "only_v_info":only_v_info,
            "heatmap_info": heatmap_info,
            "position_info":position_info,
            "monster_info":monster_info,
            "event_info":event_info,
            "rotation_info":rotation_info,
        } 
        
        return features
    
    def padding_positions(self,positions,max_nums):
        tmp = []
        num_position = []
        for position in positions:
            num = len(position)
            position = torch.as_tensor(position).reshape(-1, 2) 
            
            padding_num = max_nums - num
            position = torch.nn.functional.pad(position, (0, 0, 0, padding_num), 'constant', 0)
            tmp.append(position)
            # num_position.append(num)
            num_position.append(min(num,max_nums))
        return torch.stack(tmp,dim=0),torch.tensor(num_position)
    
    def padding_positions_with_time(self,positions,max_nums):
        tmp = []
        num_position = []
        for position in positions:
            num = len(position)
            position = torch.as_tensor(position).reshape(-1, 3) 
            
            padding_num = max_nums - num
            position = torch.nn.functional.pad(position, (0, 0, 0, padding_num), 'constant', 0)
            tmp.append(position)
            # num_position.append(num)
            num_position.append(min(num,max_nums))
        return torch.stack(tmp,dim=0),torch.tensor(num_position)
        
        
    def find_nearest_doors(self,pos):
        dis_matrix = (torch.as_tensor(pos) - self.doors_pos_matrix).norm(p=2,dim=-1)
        index = dis_matrix.le(self.doors_visble_distance)
        return self.doors_categorys[index]

      
    def get_normal_action(self, model_action, state, model_input,info):
        res = []
        state_info = info['player_state']
        own_player_state = state_info[self.id]

        action_dict = {k: v.item() for k, v in model_action.items()}

        move_dir_idx = action_dict['move_dir']
        body_action_idx = action_dict['body_action']
        yaw_idx = action_dict['yaw']
        pitch_idx = action_dict['pitch']
        body_action_type = self.BODY_ACTION_LIST[body_action_idx]

        if body_action_type == 'stop':
            move_dir_vect = Vector2(x=0, y=0)
        elif self.rel_move_dir:
            own_x = own_player_state.state.position.x
            own_y = own_player_state.state.position.y
            own_angle = math.atan2(own_y, own_x)
            move_dir_angle = self.MOVE_DIR_LIST[move_dir_idx] / 180 * math.pi + own_angle
            move_dir_vect = Vector2(x=math.cos(move_dir_angle), y= math.sin(move_dir_angle))
        else:
            move_dir_angle = self.MOVE_DIR_LIST[move_dir_idx] / 180 * math.pi
            move_dir_vect = Vector2(x=math.cos(move_dir_angle), y= math.sin(move_dir_angle))
        res.append(
            (ActionType.ACTION_MOVE, ActionMove(ai_id=self.id, direction=move_dir_vect)))

        body_action = self.get_body_action(body_action_type)
        if body_action is not None:
            res.append(body_action)

        yaw_angle = self.YAW_LIST[yaw_idx]
        pitch_angle = self.PITCH_LIST[pitch_idx]
        if self.rel_move_angle:
            rot=own_player_state.camera.rotation
            res.append((ActionType.ACTION_FOCUS, ActionFocus(ai_id=self.id, rotation=Vector3(0, max(min(rot.y+pitch_angle,80),-75), (rot.z+yaw_angle)%360))))
        else:
            res.append((ActionType.ACTION_FOCUS, ActionFocus(ai_id=self.id, rotation=Vector3(0, pitch_angle, yaw_angle))))
        # Vector3
        # float x=1; //x/roll
        # float y=2; //y/pitch
        # float z=3; //z/yaw

        return res



    def generate_random_action(self):
        action_list = self.generate_action_list()
        rsp = AIStateResponse()
        pr = PlayerResultInfo()
        pr.id = self.id
        pr.ai_action = AIAction()
        rsp.result = [pr]
        rsp.result[0].ai_action.actions = action_list
        # self.last_action = action_list
        return rsp

    def generate_action_list(self):
        action_list = []
        # action_move = ActionMove(ai_id=self.id, direction=Vector2(100, 100))
        # action_list.append(action_move)
        type, random_action = self._generate_random_action()
        am_bytes = bytes(random_action)
        action = Action(type=type, action_data=am_bytes)
        action_list.append(action)
        return action_list

    def _generate_random_action(self):
        all_action_list = [(ActionType.ACTION_FOCUS, ActionFocus(ai_id=self.id, rotation=Vector3(100, 100, 20))),
                           (ActionType.ACTION_MOVE, ActionMove(ai_id=self.id, direction=Vector2(100, 100))),
                           (ActionType.ACTION_RUN, ActionRun(ai_id=self.id, )),
                           (ActionType.ACTION_SLIDE, ActionSlide(ai_id=self.id)),
                           (ActionType.ACTION_CROUCH, ActionCrouch(ai_id=self.id)),
                           (ActionType.ACTION_DRIVE, ActionDrive(ai_id=self.id, vehicle_id=0, drive_type=2)),
                           (ActionType.ACTION_JUMP, ActionJump(ai_id=self.id)),
                           (ActionType.ACTION_GROUND, ActionGround(ai_id=self.id)),
                           (ActionType.ACTION_FIRE,
                            ActionFire(ai_id=self.id, target_position=Vector3(100, 100, 20), single=1)),
                           (ActionType.ACTION_AIM, ActionAim(ai_id=self.id)),
                           (ActionType.ACTION_SWITCH_WEAPON, ActionSwitchWeapon(ai_id=self.id, weapon_slot=0)),
                           (ActionType.ACTION_RELOAD, ActionReload(ai_id=self.id)),
                           (ActionType.ACTION_PICK, ActionPick(ai_id=self.id, item_id=0)),
                           (ActionType.ACTION_CONSUME_ITEM,
                            ActionConsumeItem(ai_id=self.id, item_id=0, category=0)),
                           (ActionType.ACTION_DROP_ITEM,
                            ActionDropItem(ai_id=self.id, item_id=0, count=1, category=0)),
                           (ActionType.ACTION_HOLD_THROWN_ITEM,
                            ActionHoldThrownItem(ai_id=self.id, thrown_id=1, category=2)),
                           (ActionType.ACTION_CANCEL_THROW, ActionCancelThrow(ai_id=self.id)),
                           (ActionType.ACTION_THROW, ActionThrow(ai_id=self.id, throw_type=0)),
                           (ActionType.ACTION_RESCUE, ActionRescue(ai_id=self.id, player_id=1)),
                           (ActionType.ACTION_DOOR_SWITCH, ActionDoorSwitch(ai_id=self.id, op=1, item_id=2)),
                           (ActionType.ACTION_SWIM, ActionSwim(ai_id=self.id, op=1)),
                           #    (ActionType.ACTION_SKILL_BTN_DOWN,ActionSkill(ai_id=self.id, skill_id=0)),
                           (ActionType.ACTION_OPEN_AIRDROP, ActionOpenAirDrop(ai_id=self.id, air_drop_id=0)), ]
        return random.choice(all_action_list)



    def get_move_dir_action(self,val,state,info):
        state_info = info['player_state']
        own_player_state = state_info[self.id]
        move_dir = (val - own_player_state.state.rotation.z)*math.pi/180
        move_dir_vect = Vector2(x=math.cos(move_dir), y= math.sin(move_dir))
        return (ActionType.ACTION_MOVE, ActionMove(ai_id=self.id, direction=move_dir_vect))
    
    def get_yaw_action(self,val,state,info):
        return (ActionType.ACTION_FOCUS, ActionFocus(ai_id=self.id, rotation=Vector3(0, self.pitch_angle, val)))
    
    def get_pitch_action(self,val,state,info):
        # print("pitch",val)
        return (ActionType.ACTION_FOCUS, ActionFocus(ai_id=self.id, rotation=Vector3(0, val, self.yaw_angle)))
    
    
    def get_pickup_action(self,info,objs):
        return self.supply_script.process_action(self.id,objs,info)
    

    def get_body_action(self,val,state,info,objs):

        player_state = info['player_state'][self.id].state

        if val == 'none':
            return None
        if val == 'run':
            if player_state.is_running == True:
                return (ActionType.ACTION_RUN, ActionRun(ai_id=self.id, op=False))
            else:
                return (ActionType.ACTION_RUN, ActionRun(ai_id=self.id, op=True))
        if val == 'slide':
            if player_state.body_state == 9:
                return (ActionType.ACTION_SLIDE, ActionSlide(ai_id=self.id, op=False))
            else:
                return (ActionType.ACTION_SLIDE, ActionSlide(ai_id=self.id, op=True))
        if val == 'crouch':
            if player_state.body_state == 2:
                return (ActionType.ACTION_CROUCH, ActionCrouch(ai_id=self.id, op=False))
            else:
                return (ActionType.ACTION_CROUCH, ActionCrouch(ai_id=self.id, op=True))
        if val == 'jump':
            return (ActionType.ACTION_JUMP, ActionJump(ai_id=self.id))
        if val == 'ground':
            if player_state.body_state == 8:
                return (ActionType.ACTION_GROUND, ActionGround(ai_id=self.id, op=False))
            else:
                return (ActionType.ACTION_GROUND, ActionGround(ai_id=self.id, op=True))
        if val == "stop":
            move_dir_vect = Vector2(x=0, y=0)
            return (ActionType.ACTION_MOVE, ActionMove(ai_id=self.id, direction=move_dir_vect))
        if val == "open_close_door":
            return self.get_door_action(info,objs) 
        if val == "rescue":
            return self.get_rescue_action()
    def get_others_action(self,val,state,info,objs,target_unit ):
        if val == 'fire_stop':
            return self.get_fire_action(info, target_unit, stop=True, adjust=False)
        elif val == "fire_stop_adjust":
            return self.get_fire_action(info,target_unit, stop=True, adjust=True)
    


        
    def get_drop_action(self,state,info,objs):
        return self.supply_script.drop_supply_actions(self.id,info)
        
    def get_items_action(self,val,state,info, objs,target_unit):
        if val == 'fire':
            # 50m 不考虑 attention
            return self.get_fire_action(info,target_unit=torch.tensor([0]))
        if val == 'reloading':
            return (ActionType.ACTION_RELOAD, ActionReload(ai_id=self.id))
        if val == "pick_up":
            return self.get_pickup_action(info , objs)
        if val == "drop_supply":
            self.get_drop_action(state,info, objs)
        if val == 'treat':
            return self.get_recover_action(info)
    
    def get_switch_weapons_action(self,val,state,info):
        return (ActionType.ACTION_SWITCH_WEAPON, ActionSwitchWeapon(ai_id=self.id, weapon_slot=val))
    
    def get_door_action(self,  info, objs):
        # =====================
        # 仅选择最近的一道门即可
        # =====================
        doors = info['doors']
        own_position = info['player_state'][self.id].state.position
        dis_matrix = (torch.as_tensor([own_position.x,own_position.y,own_position.z]) - self.doors_pos_matrix).norm(p=2,dim=-1)
        index = dis_matrix.argmin()
        door_id = self.doors_categorys[index].item()
        nearest_door = doors[door_id]
        if nearest_door.state == 0:  
            return (ActionType.ACTION_DOOR_SWITCH, ActionDoorSwitch(self.id, True, door_id))
        elif nearest_door.state == 1:  
            return (ActionType.ACTION_DOOR_SWITCH, ActionDoorSwitch(self.id, False, door_id))
        else:
            return None
    def get_recover_action(self,info):
        # ========================================
        # 不用考虑周围有人来判定是否打药，让模型自己学
        # =========================================
        if len(self.recovery_hp_items_list) == 0:
            return None
        player_states = info['player_state']
        own_player_state = player_states[self.id]
        item_category_id = [[abs(RECOVER_ITEM[bp_item.category]+ own_player_state.state.hp - 100),
                             bp_item.id]
                            for bp_item in self.recovery_hp_items_list]
        item_category_id = sorted(item_category_id)

        return (ActionType.ACTION_CONSUME_ITEM, ActionConsumeItem(self.id, item_id=item_category_id[0][1]))

            
    def get_rescue_action(self):
        if self.need_rescued_teammate_id is None:
            return None
        move_dir_vect = Vector2(x=0, y=0)
        stop = (ActionType.ACTION_MOVE, ActionMove(ai_id=self.id, direction=move_dir_vect))
        stop_and_rescure = [stop, (ActionType.ACTION_RESCUE, ActionRescue(self.id, self.need_rescued_teammate_id))]
        return stop_and_rescure
    
    def  get_fire_action(self,info, target_unit, stop = False, adjust=False):
        # ================================
        # 仅打在视野内最近的一个人即可 
        # ================================
        if target_unit is not None and len(self.visible_enemy_id_queue) > 0:
            target_enemy_id = self.visible_enemy_id_queue[target_unit[0].item()][1]
        # elif self.nearest_enemy_id is not None:
        #      target_enemy_id = self.nearest_enemy_id
        if  target_unit is None or self.activate_category == 0 or len(self.visible_enemy_id_queue) == 0:
            return None
        if not stop:
            if self.fire_no_stop_mask:
                return None
        else:
            if self.fire_stop_mask:
                return 
        
        player_states = info['player_state']
        own_player_state = player_states[self.id]
        
        own_pos = self.vector2array(own_player_state.state.position)
        own_rotation = self.vector2array(own_player_state.camera.rotation)
        target_state = player_states[target_enemy_id]
        target_pos = self.vector2array(target_state.state.position)

        if adjust:
            target_speed = self.vector2array(target_state.state.speed)
            distance = np.sqrt((own_pos[0]-target_pos[0])**2+(own_pos[1]-target_pos[1])**2+(own_pos[2]-target_pos[2])**2)
            ## 50m给定一个最低值，150m给定一个最高值，然后可以根据距离算出一个值，最后再加上一个很小的噪声
            min_val = 0.1 
            max_val = 0.3
            alpha=0.1 ##噪声扰动的程度
            ratio = (distance-5000)/(15000-5000)*(max_val - min_val) + (1+alpha*random.random())*min_val
            # ratio = 0.3
            target_pos[0],target_pos[1],target_pos[2] = target_pos[0]+ratio*target_speed[0],target_pos[1]+ratio*target_speed[1],target_pos[2]+ratio*target_speed[2]

        rel_target_pos = target_pos - own_pos
        target_rotation = self.get_roll_yaw_pitch(rel_target_pos)
        rel_target_rotation = target_rotation - own_rotation
        gun_max_dist = WEAPON_DISTANCE.get(self.activate_category, 100000)
        
        aim_action = (ActionType.ACTION_AIM, ActionAim(ai_id=self.id))
        fire_action = (ActionType.ACTION_FIRE, 
                     ActionFire(ai_id=self.id,
                                target_position=Vector3(target_pos[0], 
                                                        target_pos[1],
                                                        target_pos[2]), 
                                single=True))
        focus_action = (ActionType.ACTION_FOCUS,
                        ActionFocus(ai_id=self.id,
                                    rotation=Vector3(target_rotation[0], 
                                                    target_rotation[1], 
                                                    target_rotation[2])))
        
        
        # if abs(rel_target_rotation[1]) <= self.fire_pitch_threshold and abs(
        #         rel_target_rotation[2]) <= self.fire_yaw_threshold:
        #     acts = [fire_action,focus_action] # 近距离且已大致瞄准顺序
        #     if self.nearest_enemy_dist / gun_max_dist > self.open_aim_percent:
        #         acts = [aim_action,fire_action,focus_action] # 远距离且已大致瞄准顺序
        # else:
        #     acts = [focus_action,fire_action,] # 近距离且没有大致瞄准顺序
            
        #     if self.nearest_enemy_dist / gun_max_dist > self.open_aim_percent:
        #         acts = [focus_action,aim_action,fire_action,] # 远距离且没有大致瞄准顺序
        if abs(rel_target_rotation[1]) <= self.fire_pitch_threshold and abs(
                rel_target_rotation[2]) <= self.fire_yaw_threshold:
            
            # if self.nearest_enemy_dist / gun_max_dist > self.open_aim_percent:
            #     acts = [fire_action,focus_action] # 远距离且已大致瞄准顺序
            # else:
            acts = [fire_action,focus_action] # 近距离且已大致瞄准顺序
        else:
            # if self.nearest_enemy_dist / gun_max_dist > self.open_aim_percent:
            #     acts = [focus_action,fire_action,] # 远距离且没有大致瞄准顺序
            # else:
            acts = [focus_action,fire_action,] # 近距离且没有大致瞄准顺序
        # return acts
        if stop:
            move_dir_vect = Vector2(x=0, y=0)
            acts.append((ActionType.ACTION_MOVE, ActionMove(ai_id=self.id, direction=move_dir_vect)))
        return [acts,[self.id, target_enemy_id,stop, adjust]]
       
    def get_roll_yaw_pitch(self, pos):
        pitch = np.arctan2(pos[2], (pos[0] ** 2 + pos[1] ** 2) ** 0.5) / np.pi * 180
        yaw = np.arctan2(pos[1], pos[0]) / np.pi * 180
        # yaw = 90 - np.arctan2(pos[0], pos[1]) / np.pi * 180
        res = np.array([0, pitch, yaw])
        return res
        
    
    def vector2array(self, vect):
        if isinstance(vect, Vector2):
            vect_list = [vect.x, vect.y]
        elif isinstance(vect, Vector3):
            vect_list = [vect.x, vect.y, vect.z]
        else:
            raise NotImplementedError
        return np.array(vect_list)
    def change_speed_dir(self,action_type,action_value,info = None):
        p_state = info['player_state'][self.id].state
        self.ideal_speed_dir[:-1] = self.ideal_speed_dir[1:]
        if action_type == "move_dir":
            self.ideal_speed_dir[-1] = action_value
        elif action_type == "yaw":
            if self.ideal_speed_dir[-1] is not None:
                self.ideal_speed_dir[-1] = action_value - info['player_state'][self.id].camera.rotation.z + self.ideal_speed_dir[-1]
        elif action_type == "body_action":
            if action_value in ["stop","rescue"]:
                self.ideal_speed_dir[-1] = None
            elif action_value == "run":
                if p_state.is_running:
                    self.ideal_speed_dir[-1] = None
                else:
                    self.ideal_speed_dir[-1] = info['player_state'][self.id].camera.rotation.z
        elif  action_type == "others":
            if action_value in ["fire_stop","fire_stop_adjust"]:
                self.ideal_speed_dir[-1] = None
                    
    def transform_actions(self,model_output_action,state,info,objs,target_unit):
        # ========================================
        # 除了fire会返回list动作外，其余均为单个动作
        # ========================================
        result = []
        fire_act = None
        pick_flag = False
        for head, idx in model_output_action.items():
            if head == "target_unit":
                continue
            idx = idx.item()
            action_type, action_value = self.model_output2action[head][idx]
            self.change_speed_dir(action_type, action_value,info)
            if action_type == "move_dir":
                act = self.get_move_dir_action(action_value,state,info)
            elif action_type == "yaw":
                act = self.get_yaw_action(action_value,state,info)
            elif action_type == "pitch":
                act = self.get_pitch_action(action_value,state,info)
            elif action_type == "body_action":
                act = self.get_body_action(action_value,state,info,objs)
            elif action_type == "items_action":
                act = self.get_items_action(action_value,state,info,objs,target_unit)
                if action_value == "fire":
                    fire_act = act
                if action_value == "pick_up" or action_value=="drop_supply":
                    pick_flag = True
            elif action_type == "switch_weapons":
                act = self.get_switch_weapons_action(action_value,state,info)
            elif action_type == "others":                
                act = self.get_others_action(action_value,state,info,objs,target_unit)
                if action_value == "fire_stop" or action_value == "fire_stop_adjust":
                    fire_act = act
                
            if act is not None:
                if action_value == "fire" or action_value=="fire_stop" or action_value=="fire_stop_adjust":
                    act = act[0]
                if not isinstance(act,list):
                    result.append(act)
                else:
                    result += act
        action_list = []
        if pick_flag:
            action_list = result
        else:
            for (type, action) in result:
                am_bytes = bytes(action)
                action = Action(type=type, action_data=am_bytes)
                action_list.append(action)
        rsp = AIStateResponse()
        pr = PlayerResultInfo()
        pr.id = self.id
        pr.ai_action = AIAction()
        rsp.result = [pr]
        rsp.result[0].ai_action.actions = action_list
        return [rsp, fire_act]

def get_picth_yaw(x, y, z):
    pitch = np.arctan2(-z, (x ** 2 + y ** 2) ** 0.5) / np.pi * 180
    yaw = np.arctan2(y, x) / np.pi * 180
    return pitch, yaw


def get_torch_picth_yaw(x, y, z):
    pitch = torch.arctan2(-y, (x ** 2 + z ** 2) ** 0.5) / np.pi * 180
    yaw = torch.arctan2(x, z) / np.pi * 180
    return pitch, yaw

