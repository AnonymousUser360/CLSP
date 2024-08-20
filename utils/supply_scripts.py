import time
import os
import numpy as np
from fps.proto.ccs_ai import ActionType
from .action_utils import pick, drop, install_attachment, switch_weapon
from .supply_pickup.picker import plan,drop_bullet_attachment
from .supply_pickup.supply_filter import SupplyFilter


class SupplyModule:
    def __init__(self, ):
        self.team_num = 4
        self.player_num_per_team = 1
        self.pick_radius = 200
        self.node_step = 1 / 4 * self.pick_radius  # 1/2 * 200(拾取半径)
        self.team_step = 500
        self.stop_team_search_num = 50
        self.arrive_radius = 100
        self.map_size = [403300, 403300, 33000]
        self.filter = SupplyFilter()
        self.compress_decision_tree = True
        self.use_log = False
        if self.use_log:
            self.log_file = os.path.join(os.path.dirname(__file__), "supply_log",
                                         str(time.time()) + f"player_{self.agent_idx}" + ".txt")

    def reset(self):
        self.info = None
        self.objs = None
        self.todo_action_list = []
        self.doing_actions = []
        self.supply_item_array = None
        self.supply_vehicle_array = None
        self.supply_dist_array = None
        self.backpack_state = None
        self.last_backpack_state = None
        self.backpack_item_array = None
        self.need_replan = False
        self.own_player_state = None
        self.pick_error_counts = 0


    def process_action(self, id, objs, info, ):
        # check safety: safe continue, unsafe return
        objs = objs[id]
        self.info = info
        self.objs = objs

        player_states = info['player_state']
        self.own_player_state = own_player_state = player_states[id]

        supply_items = objs['items'] if len(objs['items']) > 0 else info['items']
        vehicle_items = objs['vehicles'] if len(objs['vehicles']) > 0 else info['vehicles']

        self.supply_item_array = self.filter.get_supplys(own_player_state, supply_items)
        self.supply_vehicle_array = self.filter.get_vehicles(own_player_state, vehicle_items)
        self.supply_dist_array = np.array([_[0] for _ in self.supply_item_array])

        self.last_backpack_state = self.backpack_state
        self.backpack_item_array, self.backpack_state = self.filter.get_backpack(own_player_state, )
        if self.last_backpack_state is None:
            self.last_backpack_state = self.backpack_state

        action_list = []
        stage_action_list = self.get_supply_actions(id, )
        action_list.extend(stage_action_list)

        return action_list


    def get_supply_actions(self, id):
        action_list = []

        nearby_supply_idx = np.where(self.supply_dist_array <= self.pick_radius)[0]
        nearby_supply_item_array = [self.supply_item_array[_] for _ in nearby_supply_idx]
        active_weapon_slot = self.info["player_state"][id].state.active_weapon_slot
        if len(nearby_supply_item_array):
            self.need_replan, self.todo_action_list = plan(nearby_supply_item_array, self.backpack_item_array,
                                                           self.backpack_state)
            if len(self.todo_action_list):
                self.doing_actions = self.todo_action_list.pop(0)
                for act in self.doing_actions:
                    action_type = act[2]
                    if action_type == ActionType.ACTION_PICK:
                        obj_id, ist_id, _, _ = act
                        action_list.append(pick(int(obj_id), int(ist_id), id))
                    elif action_type == ActionType.ACTION_DROP_ITEM:
                        ist_id, cnt, _, _ = act
                        action_list.append(drop(int(ist_id), int(cnt), id))
                    elif action_type == ActionType.ACTION_ATTACH:
                        ist_id, slot_id, _, _, attach = act
                        action_list.append(install_attachment(int(ist_id), int(slot_id), attach, id))
                    elif action_type == ActionType.ACTION_SWITCH_WEAPON:
                        slot_id, _, _, _ = act
                        if slot_id == active_weapon_slot:
                            pass
                        else:
                            action_list.append(switch_weapon(int(slot_id), id))
                    else:
                        pass
        return action_list
    def drop_supply_actions(self, id, info, ):
        player_states = info['player_state']
        own_player_state = player_states[id]

        backpack_item_array, backpack_state = self.filter.get_backpack(own_player_state, )

        action_list = []
        stage_action_list = drop_bullet_attachment(backpack_state, backpack_item_array)
        if len(stage_action_list):
            for act in stage_action_list:
                action_type = act[2]
                if action_type == ActionType.ACTION_DROP_ITEM:
                    ist_id, cnt, _, _ = act
                    action_list.append(drop(int(ist_id), int(cnt), id))
                else:
                    pass
        return action_list