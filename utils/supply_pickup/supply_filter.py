import math
from .supply_moule_item_info import all_supply_items
from .supply_type_info import SupplyType, AttachmentSubtype, WeaponSubtype


class SupplyFilter:
    def __init__(self) -> None:
        self.valid_supply_types = [_.value for _ in SupplyType]
        self.valid_attachment_types = [_.value for _ in AttachmentSubtype]
    def get_supplys(self, own_player_state, supply_items):
        # ======
        # supply
        # ======
        own_position = own_player_state.state.position
        if isinstance(supply_items, dict):
            supply_items = [v for k, v in supply_items.items()]

        supply_item_list = []
        for supply_item in supply_items:
            if isinstance(supply_item, int):
                print(supply_items)
            if supply_item.count == 0 or supply_item.attribute != 1:  # attribute 1:初始状态，0:被捡
                continue
            if supply_item.category not in all_supply_items.keys() and supply_item.category != 0:
                continue

            supply_item_pos = supply_item.position
            supply_item_category = supply_item.category
            supply_item_rel_x = supply_item_pos.x - own_position.x
            supply_item_rel_y = supply_item_pos.y - own_position.y
            supply_item_rel_z = supply_item_pos.z - own_position.z
            supply_item_obj_id = supply_item.id
            supply_not_box = len(supply_item.itemlist) == 0
            distance = math.sqrt(supply_item_rel_x ** 2 + supply_item_rel_y ** 2 +
                                 supply_item_rel_z ** 2)

            if supply_not_box:
                supply_item_ist_id = 0
                supply_item_list.append([
                    distance,
                    supply_item.count,
                    supply_item_pos.x,
                    supply_item_pos.y,
                    supply_item_pos.z,
                    supply_item_category,
                    supply_item_obj_id,
                    supply_item_ist_id,
                ])
            elif supply_item_category == 0:
                for ist in supply_item.itemlist:
                    supply_item_ist_id = ist.instance_id
                    supply_item_category = ist.category
                    supply_item_count = ist.count

                    supply_item_list.append([
                        distance,
                        supply_item_count,
                        supply_item_pos.x,
                        supply_item_pos.y,
                        supply_item_pos.z,
                        supply_item_category,
                        supply_item_obj_id,
                        supply_item_ist_id,
                    ])
        # supply_item_array = named_array(np.array(supply_item_list).reshape(-1,8), colnames=['dist', 'count', 'x','y','z','category','obj_id', 'ist_id'])
        return supply_item_list

    def get_backpack(self, own_player_state):
        # ========
        # backpack
        # ========
        backpack_items = own_player_state.backpack.backpack_item
        backpack_item_list = []

        backpack_state = {
            SupplyType.GUN: [],
            3101: 0,
            3102: 0,
            3103: 0,
            3104: 0,
            3105: 0,
            3401: 0,
            # 3402: 0,
            # 3403: 0,
            3301: 0,
            3302: 0,
            # 3303: 0,
            SupplyType.CLOTHES: -1,
            SupplyType.HELMET: -1,
            SupplyType.SHIELD: -1,
            SupplyType.BAG: -1,
            AttachmentSubtype.SILENCER: [],
            AttachmentSubtype.SCOPE: [],
            AttachmentSubtype.HANDLE: [],
            AttachmentSubtype.MAGAZINE: [],
            AttachmentSubtype.STOCK: [],
            "attachments": {},
        }

        for idx, backpack_item in enumerate(backpack_items):
            backpack_item_category = backpack_item.category
            backpack_item_type = all_supply_items[backpack_item_category].type
            backpack_item_subtype = all_supply_items[backpack_item_category].subtype

            backpack_item_id = backpack_item.id
            backpack_item_count = backpack_item.count
            backpack_item_in_use = backpack_item.in_use

            backpack_item_list.append(
                [backpack_item_category, backpack_item_id, backpack_item_count, backpack_item_in_use,
                 backpack_item_type, backpack_item_subtype])
            if backpack_item_type == SupplyType.GUN:
                backpack_item_list.pop(-1)
                continue
                backpack_state[SupplyType.GUN].append([backpack_item_category, backpack_item_id])
            elif backpack_item_type == SupplyType.BULLET:
                backpack_state[backpack_item_category] += backpack_item_count
            elif backpack_item_type ==  SupplyType.MEDICINE:
                backpack_state[backpack_item_category] += backpack_item_count
            elif backpack_item_type == SupplyType.BOMB:
                backpack_state[backpack_item_category] += backpack_item_count
            elif backpack_item_type == SupplyType.CLOTHES:
                backpack_state[SupplyType.CLOTHES] = backpack_item_category
            elif backpack_item_type == SupplyType.HELMET:
                backpack_state[SupplyType.HELMET] = backpack_item_category
            elif backpack_item_type == SupplyType.SHIELD:
                backpack_state[SupplyType.SHIELD] = backpack_item_category
            elif backpack_item_type == SupplyType.BAG:
                backpack_state[SupplyType.BAG] = backpack_item_category
            elif backpack_item_type == SupplyType.ATTACHMENT:
                if backpack_item_category % 200000000 > 6000: backpack_item_category = transfer_attachment_category(backpack_item_category)
                backpack_item_list.pop(-1)
                backpack_item_list.append(
                [backpack_item_category, backpack_item_id, backpack_item_count, backpack_item_in_use,
                 backpack_item_type, backpack_item_subtype])
                backpack_state['attachments'][backpack_item_id] = backpack_item_list[-1]
                if backpack_item_subtype in self.valid_attachment_types:
                    att_sub_type = AttachmentSubtype(backpack_item_subtype)
                    backpack_state[att_sub_type].append([backpack_item_category, backpack_item_id])
                else:
                    print(f'dont support category {backpack_item_category} type{backpack_item_type} subtype{backpack_item_subtype}!!!!!!!!')
                    raise NotImplementedError
            else:
                print(f'dont support category {backpack_item_category} type{backpack_item_type} !!!!!!!!')
                raise NotImplementedError

        weapon_items = own_player_state.weapon.player_weapon
        for weapon in weapon_items:
            weapon_category, weapon_id = weapon.category, weapon.id
            weapon_type = all_supply_items[weapon_category].type
            weapon_subtype = all_supply_items[weapon_category].subtype
            weapon_count = 1
            weapon_in_use = weapon.slot_id == own_player_state.state.active_weapon_slot
            backpack_state[SupplyType.GUN].append([weapon_category, weapon_id, weapon.attachments, weapon_in_use, weapon.slot_id])
            backpack_item_list.append([weapon_category, weapon_id, weapon_count, weapon_in_use, weapon_type, weapon_subtype])

        if len(backpack_state[AttachmentSubtype.SCOPE]) > 0:
            cate_0, cate_1 = backpack_state[AttachmentSubtype.SCOPE][0][0], backpack_state[AttachmentSubtype.SCOPE][-1][0]
            if all_supply_items[cate_0].priority > all_supply_items[cate_1].priority:
                backpack_state[AttachmentSubtype.SCOPE].reverse()
        # backpack_item_array = named_array(np.array(backpack_item_list).reshape(-1, 6),
        #                                   colnames=['category', 'id', 'count', 'in_use', 'type', 'subtype'])
        return backpack_item_list, backpack_state

    def get_vehicles(self, own_player_state, vehicle_items):
        own_position = own_player_state.state.position
        if isinstance(vehicle_items, dict):
            vehicle_items = [v for k, v in vehicle_items.items()]

        vehicle_item_list = []
        for vehicle_item in vehicle_items:
            vehicle_item_category = vehicle_item.category
            vehicle_item_pos = vehicle_item.position
            vehicle_item_rel_x = vehicle_item_pos.x - own_position.x
            vehicle_item_rel_y = vehicle_item_pos.y - own_position.y
            vehicle_item_rel_z = vehicle_item_pos.z - own_position.z
            vehicle_item_obj_id = vehicle_item.id
            distance = math.sqrt(vehicle_item_rel_x ** 2 + vehicle_item_rel_y ** 2 +
                                 vehicle_item_rel_z ** 2)
            if distance < 10000:
                vehicle_item_list.append([vehicle_item_category, vehicle_item_pos, vehicle_item_obj_id])
        return vehicle_item_list

def transfer_attachment_category(category):
    category = category % 10000 + 200000000
    return category
