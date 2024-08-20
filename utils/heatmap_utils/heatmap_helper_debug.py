import cv2
import numpy as np
from PIL import Image
import torch
# from tqdm import tqdm
import torch.nn as nn
import time
from bigrl.core.utils import EasyTimer
from bigrl.core.utils import read_config
import os



default_heatmap_config = read_config(os.path.join(os.path.dirname(__file__), 'default_heatmap_config.yaml'))





def visualize(depth_map, plt_message):
    img = np.array(depth_map[0])
    w,h = img.shape
    img = cv2.resize(img, (h*5, w*5))
    
    img = img.astype(np.uint8)
    
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)  # COLORMAP_BONE COLORMAP_JET
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    color = (255, 0, 0)  # 字体颜色
    idx = 1
    for k,v in plt_message.items():
        pos = (w // 5, idx * (h // 5))
        img = cv2.putText(img, f"{k}:{v}", pos, cv2.FONT_HERSHEY_TRIPLEX, 0.25, color)   # FONT_HERSHEY_TRIPLEX
        idx += 1
    return img

def visualize_heatmap(depth_map):
    img = np.array(depth_map[0,0])
    w,h = img.shape
    img = cv2.resize(img, (h*5, w*5))
    
    img = img.astype(np.uint8)
    
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)  # COLORMAP_BONE COLORMAP_JET
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    return img




# def load_heatmap(heat_map_path):
#     print("start to load heat map!!!!!!")
#     data = torch.load(heat_map_path)
#     share_data = {}
#     # for name,detail in tqdm(data.items()):
#     #     share_data[name] = {}
#     #     for k,v in tqdm(detail.items()):
#     #         share_data[name][k] = v.share_memory_()
#     for name,detail in data.items():
#         share_data[name] = {}
#         for k,v in detail.items():
#             if isinstance(v,dict):
#                 for k1,v1 in v.items():
#                     share_data[name][k][k1] = v1.share_memory_()
#             else:
#                 share_data[name][k] = v.share_memory_()
#     return share_data
def load_more_map():
    heat_map_delta = 20
    maxpool = nn.MaxPool2d(kernel_size=heat_map_delta, stride=heat_map_delta)
    house_map_raw = torch.load("/mnt/nfs2/bbb/bbb/fps_env_test/share_data/0529_config/house_if.torch")
    water_map_raw = torch.load("/mnt/nfs2/bbb/bbb/fps_env_test/share_data/0529_config/water_if.torch")

    house_map = house_map_raw.float().share_memory_()
    water_map = water_map_raw.float().share_memory_()


    return {
        "house_map":house_map,
        "water_map":water_map,
        "house_map_max_pool":maxpool(house_map_raw.unsqueeze(dim=0).float())[0].share_memory_(),
        "water_map_max_pool":maxpool(water_map_raw.unsqueeze(dim=0).float())[0].share_memory_(),
    }

class MoreMap():
    def __init__(self,more_map, cfg) -> None:
        self.more_map = more_map
        self.cfg = cfg
        
        self.scale = 2
        self.heat_map_size = [640 * self.scale, 640 * self.scale]
        self.pool_map_size = [(640 * self.scale)//2, (640 * self.scale)//2]
        self.heat_map_center = [(self.heat_map_size[0] )//(20 *self.scale),
                                (self.heat_map_size[1] )//(20 * self.scale)]
        self.maxpool = nn.MaxPool2d(kernel_size=10*self.scale, stride=10*self.scale)
        self.eval_mode = self.cfg.actor.get("eval_mode",False) == True
        if self.eval_mode:
            self.visdom_map = {
                "water":{},
                "house":{},
            }

    def generate_map(self,position_info, get_target_pos = False):
        
        position_x, position_y, = position_info["position_x"],position_info["position_y"]
         # 计算出最近的水和房区的位置
        nearest_water = None
        nearest_house = None
        if get_target_pos:
            xb = (position_x-self.pool_map_size[0])
            yb = (position_y-self.pool_map_size[1])
            xt = (position_x+self.pool_map_size[0])
            yt = (position_y+self.pool_map_size[1])
            water_map = self.more_map["water_map"][xb:xt, yb:yt] / 1.0
            house_map = self.more_map["house_map"][xb:xt, yb:yt] / 1.0
            water_map = self.padding_heatmap(water_map)
            house_map = self.padding_heatmap(house_map)
            if get_target_pos:
                
                given_index = self.pool_map_size
                nonzero_indices = torch.nonzero(water_map)

                if len(nonzero_indices):
                    distances = torch.abs(nonzero_indices - torch.tensor(given_index)).sum(dim=1)
                    nearest_index = nonzero_indices[distances.argmin()]

                    nearest_water = [
                        (position_x + nearest_index[0] - given_index[0]) * 10,
                        (position_y + nearest_index[1] - given_index[1]) * 10,
                    ]

                nonzero_indices = torch.nonzero(house_map)
                if len(nonzero_indices):
                    distances = torch.abs(nonzero_indices - torch.tensor(given_index)).sum(dim=1)
                    nearest_index = nonzero_indices[distances.argmin()]

                    nearest_house = [
                        (position_x + nearest_index[0] - given_index[0])*10,
                        (position_y + nearest_index[1] - given_index[1])*10,
                    ]


            water_map = self.maxpool(water_map.unsqueeze(dim=0))[0]
            
            house_map = self.maxpool(house_map.unsqueeze(dim=0))[0]
        else:
            xb = (position_x-self.pool_map_size[0])//20
            yb = (position_y-self.pool_map_size[1])//20
            xt = (position_x+self.pool_map_size[0])//20
            yt = (position_y+self.pool_map_size[1])//20
            water_map = self.more_map["water_map_max_pool"][xb:xt, yb:yt] / 1.0
            house_map = self.more_map["house_map_max_pool"][xb:xt, yb:yt] / 1.0
            water_map = self.padding_heatmap_max_pool(water_map)
            house_map = self.padding_heatmap_max_pool(house_map)

        house_map[self.heat_map_center[0] -1 :self.heat_map_center[0] +1,
                self.heat_map_center[1] -1:self.heat_map_center[1]+1] = 2
        water_map[self.heat_map_center[0] -1 :self.heat_map_center[0] +1,
                self.heat_map_center[1] -1:self.heat_map_center[1]+1] = 2
        guide_map = torch.stack([house_map, water_map ],dim=-3)

        if self.eval_mode:

            self.visdom_map["water"] = water_map
            self.visdom_map["house"] = house_map
        return  guide_map, nearest_house, nearest_water
    
    
    def check_pos(self,position_info):
        position_x, position_y, = position_info["position_x"],position_info["position_y"]
        xb = (position_x-5)
        yb = (position_y-5)
        xt = (position_x+5)
        yt = (position_y+5)
        water_map = self.more_map["water_map"][xb:xt, yb:yt]
        house_map = self.more_map["house_map"][xb:xt, yb:yt]
        water_if = water_map.sum() >= 80
        house_if = house_map.sum() >= 80
        return water_if, house_if

    def padding_heatmap_max_pool(self,heat_map):
        heat_map_size = [64,64]
        if heat_map.shape[-1] != heat_map_size[-1] or heat_map.shape[-2] != heat_map_size[-2]:
            heat_map = torch.nn.functional.pad(heat_map, (0, heat_map_size[-1]-heat_map.shape[-1], 0, heat_map_size[-2]-heat_map.shape[-2]), 'constant', 0)
        return heat_map
    
    def padding_heatmap(self,heat_map): 
        if heat_map.shape[-1] != self.heat_map_size[-1] or heat_map.shape[-2] != self.heat_map_size[-2]:
            heat_map = torch.nn.functional.pad(heat_map, (0, self.heat_map_size[-1]-heat_map.shape[-1], 0, self.heat_map_size[-2]-heat_map.shape[-2]), 'constant', 0)
        return heat_map
    
    def generate_heat_map_trajectory(self,position_info_traj):
        trj_maps = []
        for trj_idx in range(position_info_traj["position_x"].shape[0]):
            position_info = {
            "position_x":position_info_traj["position_x"][trj_idx],
            "position_y":position_info_traj["position_y"][trj_idx],
            }
            heat_maps = self.generate_map(position_info)[0]
            trj_maps.append(heat_maps)
        return torch.stack(trj_maps,dim=0)


def load_heatmap(heat_map_path):
    print("start to load heat map!!!!!!")
    data = torch.load(heat_map_path)
    share_data = {
        "server_0913":{

        }
    }
    for name,detail in data.items():
        share_data[name] = {}
        for k,v in detail.items():
            if isinstance(v,dict):
                share_data[name][k] = {}
                for k1,v1 in v.items():
                    share_data[name][k][k1] = v1.share_memory_()
            else:
                share_data[name][k] = v.share_memory_()
    server_0913 = torch.load("/mnt/nfs2/bbb/bbb/fps_env_test/share_data/0913_config/bev_0913.pt")
    for name,detail in server_0913.items():
        share_data["server_0913"][name] = {}
        for k,v in detail.items():
            if isinstance(v,dict):
                share_data["server_0913"][name][k] = {}
                for k1,v1 in v.items():
                    share_data["server_0913"][name][k][k1] = v1.share_memory_()
            else:
                share_data["server_0913"][name][k] = v.share_memory_()
    
    return share_data

class HeatMap():
    def __init__(self,cfg,heat_map_data,variable_record = None,loc = "actor"):
        self.loc = loc
        self.variable_record = variable_record
        self.cfg = cfg
        # self.timer = EasyTimer(cuda=False)
        self.heat_map_data = heat_map_data
        self.heat_map_loc_norm = self.cfg.env.heat_map.heat_map_loc_norm
        self.heat_map_scale = self.cfg.env.heat_map.heat_map_scale
        self.heat_map_size = self.cfg.env.heat_map.heat_map_size
        self.heat_map_deltas = self.cfg.env.heat_map.heat_map_deltas
        self.all_pool_map_size = {}
        self.heat_map_center = [(self.heat_map_size[0] )//2,
                                (self.heat_map_size[1] )//2]
        self.map_3d_height_near_tol = 100
        
        self.height_delta = [150,300,450,600,750,900]

        self.unroll_len = self.cfg.learner.data.unroll_len
        for k,v in self.heat_map_deltas.items():
            if k != "bottom2top_10":
                for delta in v:
                    if delta not in self.all_pool_map_size.keys():
                        self.all_pool_map_size[delta] = [(delta * self.heat_map_size[0])//2,
                                                        (delta * self.heat_map_size[1])//2]
        self.eval_mode = self.cfg.actor.get("eval_mode",False) == True
        if self.eval_mode:
            self.visdom_map = {
                "top2bottom_10":{},
                "pitch_delta_10":{},
                "yaw_delta_10":{},
                "history_position":{},
                "height": {
                    "ground":0,
                    "delta":0,
                    "z":0,
                }
            }
        self.setup_index_info()
    
    def setup_index_info(self):
        self.expand =  self.cfg.env.heat_map.get("edge_expand",32)
        norm = 40
        
        self.heights = {
            "wild_area": [50 * (_+1) for _ in range(default_heatmap_config.heat_map.wild_area.height_nums)],
            "server_0913": [50 * (_+1) for _ in range(40)],
        }
        self.housing_area_names = []
        self.housing_area_center = []
        self.housing_area_radius = []
        self.bottom_left_coord = {}
        for k,v in default_heatmap_config.heat_map.housing_area.items():
            self.housing_area_names.append(k)
            self.housing_area_center.append([v.center[0]//self.heat_map_scale , v.center[1]//self.heat_map_scale] )
            self.housing_area_radius.append(v.radius//self.heat_map_scale )
            self.bottom_left_coord[k] = [int(v.center[0]//norm-v.radius//norm - self.expand),
                                         int(v.center[1]//norm-v.radius//norm - self.expand),
                                         ]
            self.heights[k] = [50 * (_+1) for _ in range(v.height_nums)]
        self.housing_area_center = torch.tensor(self.housing_area_center)
        self.housing_area_radius = torch.tensor(self.housing_area_radius)

    def find_heat_map_layer(self,x,y):
        coord = torch.tensor([x, y])
        bool_tensor = (((self.housing_area_center - coord)**2).sum(dim=1).sqrt())  <  self.housing_area_radius
        
        if len(torch.nonzero(bool_tensor)[:, 0]) == 0:
            return "wild_area"
        else:
            return self.housing_area_names[torch.nonzero(bool_tensor)[:, 0][0].item()]


    def padding_heatmap(self,heat_map):
        if heat_map.shape[-1] != self.heat_map_size[-1] or heat_map.shape[-2] != self.heat_map_size[-2]:
            heat_map = torch.nn.functional.pad(heat_map, (0, self.heat_map_size[-1]-heat_map.shape[-1], 0, self.heat_map_size[-2]-heat_map.shape[-2]), 'constant', 0)
        return heat_map
    def generate_heat_map(self,position_info,server_is_0913 = False):
        server_name = "server_0913" if server_is_0913 else  None
        
        position_x, position_y,position_z = position_info["position_x"],position_info["position_y"],position_info["position_z"]
        
        ## 先找出是否为房区，且在哪一个房区
        if server_name == "server_0913":
            layer_name = server_name
        else:
            layer_name = self.find_heat_map_layer(position_x,position_y,)

        top2bottom_teammate_position = position_info["top2bottom_teammate_position"]
        top2bottom_enemy_position = position_info["top2bottom_enemy_position"]

        yaw_teammate_position = position_info["yaw_teammate_position"]
        yaw_enemy_position = position_info["yaw_enemy_position"]

        yaw_door_position = position_info["yaw_door_position"]
        
        top2bottom_teammate_nums = position_info["top2bottom_teammate_nums"]
        top2bottom_enemy_nums = position_info["top2bottom_enemy_nums"]

        yaw_door_nums = position_info["yaw_door_nums"]
        yaw_teammate_nums = position_info["yaw_teammate_nums"]
        yaw_enemy_nums = position_info["yaw_enemy_nums"]
        heat_maps = []
        for name in ["top2bottom_10","pitch_delta_10","yaw_delta_10"]:
            start = time.time()
            deltas = self.heat_map_deltas[name]
            if name == "top2bottom_10":
                
                for idx in range(len(deltas)):
                    heat_map_delta = deltas[idx]
                    pool_map_size = self.all_pool_map_size[heat_map_delta]
                    xb = (position_x-pool_map_size[0])//heat_map_delta
                    yb = (position_y-pool_map_size[1])//heat_map_delta
                    xt = (position_x+pool_map_size[0])//heat_map_delta
                    yt = (position_y+pool_map_size[1])//heat_map_delta
                    if server_name == "server_0913":
                        heat_map = self.heat_map_data[server_name][name][heat_map_delta][xb:xt,
                                                                        yb:yt]
                    else:
                        heat_map = self.heat_map_data[name][heat_map_delta][xb:xt,
                                                                        yb:yt]
                    if self.eval_mode:
                        visdom_map = {"海拔图":torch.clone(heat_map)}
                    teammate_position = top2bottom_teammate_position[idx]
                    enemy_position = top2bottom_enemy_position[idx]
                    heat_map_norm = heat_map[self.heat_map_center[0] -1 :self.heat_map_center[0] +1,
                                    self.heat_map_center[1] - 1:self.heat_map_center[1]+ 1].mean()
                    heat_map = ((heat_map - heat_map_norm)/self.heat_map_loc_norm).clamp(min=-1,max = 1)
                    heat_map = self.add_people(heat_map,teammate_position,heat_map_delta,xb,yb,xt,yt,1.5,top2bottom_teammate_nums[idx])
                    heat_map = self.add_people_with_time(heat_map,enemy_position,heat_map_delta,xb,yb,xt,yt,-1.5,top2bottom_enemy_nums[idx])
                    heat_map[self.heat_map_center[0] -1 :self.heat_map_center[0] +1,
                                self.heat_map_center[1] -1:self.heat_map_center[1]+1] = 2
                    # 防止出现边界情况！！！
                    heat_map = self.padding_heatmap(heat_map)
                    if self.eval_mode:
                        visdom_map["模型海拔图"] = heat_map
                        self.visdom_map[name][heat_map_delta] = visdom_map
                    heat_maps.append(heat_map)

             
            elif name == "yaw_delta_10":
                for idx in range(len(deltas)):
                    heat_map_delta = deltas[idx]
                    if server_name == "server_0913":
                        cur_pos_ground_height = self.heat_map_data[server_name]["bottom2top_10"][heat_map_delta][position_x//heat_map_delta -1 :position_x//heat_map_delta +1,
                                                                                    position_y//heat_map_delta -1:position_y//heat_map_delta+1].mean()
                        height_delta = (cur_pos_ground_height + torch.tensor(  self.heights[server_name]) - position_z).abs()
                    else:
                        cur_pos_ground_height = self.heat_map_data["bottom2top_10"][heat_map_delta][position_x//heat_map_delta -1 :position_x//heat_map_delta +1,
                                                                                position_y//heat_map_delta -1:position_y//heat_map_delta+1].mean()

                        height_delta = (cur_pos_ground_height + torch.tensor(  self.heights[layer_name]) - position_z).abs()
                    if self.eval_mode:
                        self.visdom_map["height"] =  {
                                        "ground":cur_pos_ground_height.item(),
                                        "delta":height_delta.min().item(),
                                        "z":position_z.item(),
                                    }
                    if height_delta.min() > self.map_3d_height_near_tol:
                        heat_map = torch.zeros(self.heat_map_size,dtype=torch.float)
                        # heat_map_leg = torch.zeros(self.heat_map_size,dtype=torch.float)
                        if self.eval_mode:
                            visdom_map = {}
                            near_idx = -1

                    else:
                        # 要比z坐标高100cm
                        
                        near_idx = min(height_delta.argmin()+2, len( self.heights[layer_name])-1)
                        pool_map_size = self.all_pool_map_size[heat_map_delta]
                        if layer_name in ["wild_area", "server_0913"]:
                            xb = (position_x-pool_map_size[0])//heat_map_delta
                            yb = (position_y-pool_map_size[1])//heat_map_delta
                            xt = (position_x+pool_map_size[0])//heat_map_delta
                            yt = (position_y+pool_map_size[1])//heat_map_delta
                        else:
                            xb = (position_x-pool_map_size[0])//heat_map_delta - self.bottom_left_coord[layer_name][0]
                            yb = (position_y-pool_map_size[1])//heat_map_delta - self.bottom_left_coord[layer_name][1]
                            xt = (position_x+pool_map_size[0])//heat_map_delta - self.bottom_left_coord[layer_name][0]
                            yt = (position_y+pool_map_size[1])//heat_map_delta - self.bottom_left_coord[layer_name][1]
                        if server_name == "server_0913":
                            heat_map = self.heat_map_data["server_0913"][name][heat_map_delta][near_idx,xb:xt,
                                                                            yb:yt]
                        else:
                            heat_map = self.heat_map_data[name][heat_map_delta][layer_name][near_idx,xb:xt,
                                                                            yb:yt]

                        teammate_position = yaw_teammate_position[idx]
                        enemy_position = yaw_enemy_position[idx]
                        door_position = yaw_door_position[idx]

                        heat_map = heat_map/1.0

                        if self.eval_mode:
                            visdom_map = {"水平推图":torch.clone(heat_map)}

                        heat_map = self.add_people(heat_map,teammate_position,heat_map_delta,xb,yb,xt,yt,1.5,yaw_teammate_nums[idx],layer_name)
                        heat_map = self.add_people_with_time(heat_map,enemy_position,heat_map_delta,xb,yb,xt,yt,-1.5,yaw_enemy_nums[idx],layer_name)
                        heat_map = self.add_door(heat_map,door_position,heat_map_delta,xb,yb,xt,yt,yaw_door_nums[idx],layer_name)

                    heat_map[self.heat_map_center[0] -1 :self.heat_map_center[0] +1,
                            self.heat_map_center[1] -1:self.heat_map_center[1]+1] = 2

                    heat_map = self.padding_heatmap(heat_map)

                    if self.eval_mode:
                        visdom_map["模型水平推图"]=heat_map
                        self.visdom_map[name][heat_map_delta] = visdom_map

                    heat_maps.append(heat_map)
            if self.loc == "learner":
                self.variable_record.update_var(
                                        {f'learner_time/get_heatmap_{name}': time.time()- start})
                    
        return torch.stack(heat_maps,dim=-3)
    
    
    
    def add_history_position_channel(self,position_x, position_y,history_position,
                                     top2bottom_teammate_position,top2bottom_teammate_nums,
                                     top2bottom_enemy_position,top2bottom_enemy_nums,
                                     heat_map_delta=40):
        deltas = self.heat_map_deltas["top2bottom_10"]
        idx = deltas.index(heat_map_delta)
        pool_map_size = self.all_pool_map_size[heat_map_delta]
        xb = (position_x-pool_map_size[0])//heat_map_delta
        yb = (position_y-pool_map_size[1])//heat_map_delta
        xt = (position_x+pool_map_size[0])//heat_map_delta
        yt = (position_y+pool_map_size[1])//heat_map_delta
        heat_map = torch.zeros(self.heat_map_size,dtype=torch.float)
        # t = time.time()
        heat_map = self.add_history_path(heat_map,history_position,heat_map_delta,xb,yb,xt,yt)
        # print("add_history_path:",time.time()-t)
        heat_map = self.add_people(heat_map,top2bottom_teammate_position[idx],heat_map_delta,xb,yb,xt,yt,1.5,top2bottom_teammate_nums[idx])
        heat_map = self.add_people(heat_map,top2bottom_enemy_position[idx],heat_map_delta,xb,yb,xt,yt,-1.5,top2bottom_enemy_nums[idx])
        heat_map[self.heat_map_center[0] -1 :self.heat_map_center[0] +1,
                            self.heat_map_center[1] -1:self.heat_map_center[1]+1] = 2
        if self.eval_mode:
            self.visdom_map["history_position"][heat_map_delta] = {
                "轨迹图":heat_map
            }
        return heat_map
        
    def add_history_path_cp(self,heat_map,history_position,delta,xb,yb,xt,yt):
        b = time.time()
        idx = 0
        for x,y in history_position:
            x = x // delta
            y = y // delta
            if x >= xb and x<= xt and y >= yb and y <= yt:
                x1 = max(0,x-1 - xb)
                x2 = min(x+1 - xb, xt - xb - 1 )
                y1 = max(0,y-1 - yb)
                y2 = min(y+1 - yb, yt - yb - 1 )
                heat_map[x1:x2,y1:y2] = 0.5 + idx * (0.5/ 600)
            idx += 1
        print(time.time() - b)
        return heat_map
    
    def add_history_path(self,heat_map,history_position,delta,xb,yb,xt,yt):
        pos = history_position // delta
        mask = ((pos[:,0] >= xb) &(pos[:,0]< xt) & (pos[:,1] >= yb) & (pos[:,1] < yt))
        pos[:,0] = pos[:,0] - xb
        pos[:,1] = pos[:,1] - yb
        val = torch.arange(0,pos.shape[0]) * (0.5/600) + 0.5 
        pos = pos[mask]
        val = val[mask]
        heat_map = heat_map.index_put((pos[:,0],pos[:,1]),val)
        return heat_map
            
                           
    def add_people(self,heat_map,positions,delta,xb,yb,xt,yt,val,nums,layer_name = "wild_area"):
        for i in range(nums):
            x,y = positions[i][0],positions[i][1]
            if layer_name in ["wild_area", "server_0913"]:
                x = x // delta
                y = y // delta
            else:
                x = x // delta - self.bottom_left_coord[layer_name][0]
                y = y // delta - self.bottom_left_coord[layer_name][1]
            x1 = max(0,x-1 - xb)
            x2 = min(x+1 - xb, xt - xb - 1 )
            y1 = max(0,y-1 - yb)
            y2 = min(y+1 - yb, yt - yb - 1 )
            heat_map[x1:x2,y1:y2] = val
        return heat_map
    def add_people_with_time(self,heat_map,positions,delta,xb,yb,xt,yt,val,nums,layer_name = "wild_area"):
        for i in range(nums):
            x,y = positions[i][0],positions[i][1]
            if layer_name in ["wild_area", "server_0913"]:
                x = x // delta
                y = y // delta
            else:
                x = x // delta - self.bottom_left_coord[layer_name][0]
                y = y // delta - self.bottom_left_coord[layer_name][1]
            x1 = max(0,x-1 - xb)
            x2 = min(x+1 - xb, xt - xb - 1 )
            y1 = max(0,y-1 - yb)
            y2 = min(y+1 - yb, yt - yb - 1 )
            heat_map[x1:x2,y1:y2] = val + min(0.4,positions[i][2]/25)
        return heat_map
    
    def add_door(self,heat_map,positions,delta,xb,yb,xt,yt,nums,layer_name = "wild_area"):
        for i in range(nums):
            x,y,door_state = positions[i][0],positions[i][1],positions[i][2]
            if layer_name in ["wild_area", "server_0913"]:
                x = x // delta
                y = y // delta
            else:
                x = x // delta - self.bottom_left_coord[layer_name][0]
                y = y // delta - self.bottom_left_coord[layer_name][1]
            x1 = max(0,x-1 - xb)
            x2 = min(x+1 - xb, xt - xb - 1 )
            y1 = max(0,y-1 - yb)
            y2 = min(y+1 - yb, yt - yb - 1 )
            if door_state==0:
                heat_map[x1:x2,y1:y2] = 0.5
            elif door_state==1:
                heat_map[x1:x2,y1:y2] = -0.5
        return heat_map
                    
    def generate_heat_map_trajectory(self,position_info_traj):
        trj_maps = []
        for trj_idx in range(position_info_traj["position_x"].shape[0]):
            position_info = {
            "position_x":position_info_traj["position_x"][trj_idx],
            "position_y":position_info_traj["position_y"][trj_idx],
            "position_z":position_info_traj["position_z"][trj_idx],
            "yaw_door_position":position_info_traj["yaw_door_position"][trj_idx],
            "top2bottom_teammate_position":position_info_traj["top2bottom_teammate_position"][trj_idx],
            "top2bottom_enemy_position":position_info_traj["top2bottom_enemy_position"][trj_idx],
            
            "pitch_teammate_position":position_info_traj["pitch_teammate_position"][trj_idx],
            "pitch_enemy_position": position_info_traj["pitch_enemy_position"][trj_idx],
            "yaw_teammate_position":position_info_traj["yaw_teammate_position"][trj_idx],
            "yaw_enemy_position": position_info_traj["yaw_enemy_position"][trj_idx],


            "yaw_door_nums":position_info_traj["yaw_door_nums"][trj_idx],
            "top2bottom_teammate_nums":position_info_traj["top2bottom_teammate_nums"][trj_idx],
            "top2bottom_enemy_nums":position_info_traj["top2bottom_enemy_nums"][trj_idx],
            "pitch_teammate_nums":position_info_traj["pitch_teammate_nums"][trj_idx],
            "pitch_enemy_nums":position_info_traj["pitch_enemy_nums"][trj_idx],
            "yaw_teammate_nums":position_info_traj["yaw_teammate_nums"][trj_idx],
            "yaw_enemy_nums":position_info_traj["yaw_enemy_nums"][trj_idx],
            "history_position":position_info_traj["history_position"][trj_idx]
            }
            heat_maps = self.generate_heat_map(position_info,server_is_0913 = position_info_traj["server_is_0913"][trj_idx])
            trj_maps.append(heat_maps)
        return torch.stack(trj_maps,dim=0)
            
