# -*- coding:utf-8 -*
import math

from ..utils import compute_list_distance


class SupplyPlaner:
    def __init__(self, node_step=50, team_step=500, car_step=500, weight_z=1, stop_team_search_num=50):
        self.node_step = node_step
        self.team_step = team_step
        self.car_step = car_step
        self.stop_team_search_num = stop_team_search_num
        self.weight_z = weight_z

    def reset(self):
        self.searched_nodes = set()
        self.all_team_nodes = {}
        self.unsearch_nodes = {}
        self.ignored_nodes = {}

    def add_unsearch_nodes(self, supply_item_array, pid, player_num):
        node_list = []
        if len(self.searched_nodes) == 0: #  init team search nodes
            for idx, sup in enumerate(supply_item_array):
                dist, count, x, y, z, category, obj_id, ist_id = sup
                pos = (x, y, z)
                node = pos2node(pos=pos, node_step=self.team_step)
                if node not in node_list:
                    node_list.append(node)
            
            node_list.sort()
            own_node_list = []
            for idx, node in enumerate(node_list):
                if idx % player_num == pid % player_num and node not in self.unsearch_nodes.keys():
                    own_node_list.append(node)

            for idx, sup in enumerate(supply_item_array): #  ignore team constraints
                dist, count, x, y, z, category, obj_id, ist_id = sup
                pos = (x, y, z)
                node = pos2node(pos=pos, node_step=self.node_step)
                team_node = pos2node(pos=pos, node_step=self.team_step)
                if team_node in own_node_list and node not in self.unsearch_nodes.keys():
                    self.unsearch_nodes[node] = pos

        elif len(self.searched_nodes) < self.stop_team_search_num: #  keep searching in own nodes
            pass
        elif len(self.searched_nodes) >= self.stop_team_search_num:
            for idx, sup in enumerate(supply_item_array): #  ignore team constraints
                dist, count, x, y, z, category, obj_id, ist_id = sup
                pos = (x, y, z)
                node = pos2node(pos=pos, node_step=self.node_step)
                if node not in self.searched_nodes and node not in self.unsearch_nodes.keys():
                    self.unsearch_nodes[node] = pos

    def add_searched_node(self, pos, height_bias):
        node_pos = (pos[0], pos[1], pos[2] - height_bias)
        node = pos2node(node_pos, node_step=self.node_step)
        if node in self.searched_nodes:
            assert node not in self.unsearch_nodes, f"searched node {node} can't in unsearched nodes"
            print(f"node {node} is already searched!!!!!!!!!!!")
        else:
            self.searched_nodes.add(node,)
            self.unsearch_nodes.pop(node,None)

    def add_ignored_node(self, pos, height_bias):
        node_pos = (pos[0], pos[1], pos[2] - height_bias)
        node = pos2node(node_pos, node_step=self.node_step)
        if node in self.searched_nodes:
            assert node not in self.unsearch_nodes, f"searched node {node} can't in unsearched nodes"
            print(f"node {node} is already searched!!!!!!!!!!!")
        else:
            self.searched_nodes.add(node,)
            self.unsearch_nodes.pop(node,None)
            self.ignored_nodes[node] = pos

    def choose_moving_target(self, own_position, vehicle_items, height_bias):
        min_dist = float('inf')
        moving_target = None
        own_foot_position = (own_position[0], own_position[1], own_position[2] - height_bias)
        for node, node_pos in self.unsearch_nodes.items():
            # node_pos = node2pos(node, node_step=self.node_step)
            node_dist = compute_list_distance(node_pos, own_foot_position, weight=[1,1,self.weight_z])
            is_valid = True
            for vehicle in vehicle_items:
                category, vehicle_pos, obj_id = vehicle
                vehicle_pos = [vehicle_pos.x, vehicle_pos.y, vehicle_pos.z]
                node_vehicle_dist = compute_list_distance(node_pos, vehicle_pos)
                if node_vehicle_dist < self.car_step:
                    is_valid = False
                    break
            if is_valid and node_dist < min_dist:
                min_dist = node_dist
                moving_target = (node_pos[0], node_pos[1], node_pos[2] + height_bias)
        return moving_target

    def get_stat(self):
        stat = {
            'searched_nodes': len(self.searched_nodes),
            'unsearch_nodes': len(self.unsearch_nodes),
            'ignored_nodes':  len(self.ignored_nodes),
        }
        return stat


def pos2node(pos, node_step):
    x, y, z = pos
    node = (math.ceil(x / node_step), math.ceil(y / node_step), math.ceil(z / node_step))
    return node


def node2pos(node, node_step):
    x, y, z = node
    pos = (x * node_step, y * node_step, z * node_step)
    return pos
