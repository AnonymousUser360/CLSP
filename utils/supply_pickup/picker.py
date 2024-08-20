from collections import defaultdict
from fps.proto.ccs_ai import ActionType
try:
    from .supply_moule_item_info import all_supply_items, bag_level, bullet2gun
    from .supply_type_info import SupplyType, AttachmentSubtype
except:
    from supply_moule_item_info import all_supply_items, bag_level, bullet2gun
    from supply_type_info import SupplyType, AttachmentSubtype


items_has_priority = [
    SupplyType.GUN, SupplyType.BAG, SupplyType.CLOTHES,
    SupplyType.HELMET, SupplyType.SHIELD, SupplyType.ATTACHMENT    
]
attachments_special = [
    AttachmentSubtype.SCOPE
]
attachments_non_special = [
    AttachmentSubtype.MAGAZINE, AttachmentSubtype.STOCK, AttachmentSubtype.SILENCER, AttachmentSubtype.HANDLE
]


def plan(env_supplys, player_supplys, state):
    env_state = split_supplys(env_supplys, state)

    actions = plan_equip_actions(env_supplys, env_state, state)
    if len(actions) > 0: actions = [actions]
    change_gun, gun_actions = plan_gun(env_supplys, env_state, state)
    if change_gun: 
        # actions.append(gun_actions)
        actions.extend(gun_actions)
        return change_gun, actions

    change_bag, bag_actions = plan_bag(env_supplys, env_state, state)
    if change_bag: 
        actions.append(bag_actions)
        return change_bag, actions

    action_seq = plan_single_item(env_supplys, env_state, state)
    if len(action_seq): actions.append(action_seq)
    action_seq = plan_attachments(env_supplys, env_state, state)
    if len(action_seq): actions.append(action_seq)
    action_seq = plan_bullet(env_supplys, env_state, state)
    if len(action_seq): actions.append(action_seq)
    action_seq = plan_multiple_item(env_supplys, env_state, state)
    if len(action_seq): actions.append(action_seq)
    return False, actions

def plan_equip_actions(env_supplys, env_state, state):
    actions_seq = []
    # equip attachments, there is no attachments in the bag by default
    if len(state[SupplyType.GUN]) == 0: return actions_seq
    for key, att_item in state['attachments'].items():
        p_cate, pid, p_cnt, inuse, p_main_type, p_subtype = att_item
        if not inuse:
            att_useful = False
            for i, gun in enumerate(state[SupplyType.GUN]):
                gun_cate, gun_id, gun_att, gun_inuse, gun_slot = gun
                if p_cate not in all_supply_items[gun_cate].attachment: continue
                att_subtypes = [all_supply_items[state['attachments'][_][0]].subtype for _ in gun_att]
                if p_subtype not in att_subtypes: # no attachment of this type, equip
                    actions_seq.append([pid, i + 1, ActionType.ACTION_ATTACH, p_cate, True])
                    att_useful = True
                    break
                else:
                    for j, att_ist_id in enumerate(gun_att):
                        if att_subtypes[j] == p_subtype:
                            curr_att_cate = state['attachments'][att_ist_id][0]
                            if all_supply_items[curr_att_cate].priority < all_supply_items[
                                p_cate].priority:  # better attachment, equip
                                actions_seq.append(
                                    [att_ist_id, i + 1, ActionType.ACTION_ATTACH, curr_att_cate, False])
                                actions_seq.append([pid, i + 1, ActionType.ACTION_ATTACH, p_cate, True])
                                att_useful = True
                                break
                    if att_useful: break
            if not att_useful:  # unuseful attachment, drop
                # actions_seq.append([pid, i + 1, ActionType.ACTION_ATTACH, curr_att_cate, False])
                actions_seq.append([pid, p_cnt, ActionType.ACTION_DROP_ITEM, p_cate])
    return actions_seq

def plan_gun(env_supplys, env_state, state):
    actions_seq = []
    snipe_cnt = 0
    for my_gun in state[SupplyType.GUN]:  ## 判断有几把狙击枪
        snipe_cnt += my_gun[0] in [2201, 2202, 2203]  

    change_gun = len(state[SupplyType.GUN]) < 2 and len(env_state[SupplyType.GUN]) > 0
    drop_slot = 0
    pick_idx = 0

    for idx, my_gun in enumerate(state[SupplyType.GUN]):
        gun_category, gun_id, gun_att, gun_inuse, gun_slot = my_gun
        if len(env_state[SupplyType.GUN]) > 0:
            if snipe_cnt == 0:
                for i_gun, env_gun in enumerate(env_state[SupplyType.GUN]):
                    change_gun = greater(env_gun[0], gun_category)
                    pick_idx = i_gun
                    if change_gun: break
            elif snipe_cnt == 1:
                env_gun = env_state[SupplyType.GUN][0]
                if env_gun[0] not in [2201, 2202, 2203]:
                    if len(state[SupplyType.GUN]) == 1:
                        change_gun = True
                        pick_idx = 0
                    else:
                        change_gun = greater(env_gun[0], gun_category)
                        pick_idx = 0
                elif len(env_state[SupplyType.GUN]) == 1:
                    change_gun = gun_category in [2201, 2202, 2203] and greater(env_gun[0], gun_category)
                    pick_idx = 0
                else:
                    env_gun = env_state[SupplyType.GUN][-1]
                    change_gun = gun_category in [2201, 2202, 2203] and greater(env_gun[0], gun_category)
                    pick_idx = -1
            elif env_state[SupplyType.GUN][0][0] not in [2201, 2202, 2203]:
                change_gun = True
                pick_idx = 0
            elif greater(env_state[SupplyType.GUN][-1][0], gun_category):
                change_gun = True
                pick_idx = -1
            else:
                change_gun = False
        else:
            change_gun = False

        if change_gun and len(state[SupplyType.GUN]) > 1: 
            drop_slot = idx + 1
            break
        
    if change_gun:
        tmp = []
        if drop_slot > 0: # need drop items
            gun_category, gun_id, gun_att, gun_inuse, gun_slot = state[SupplyType.GUN][drop_slot - 1]
            # actions_seq.append([[drop_slot, 1, ActionType.ACTION_SWITCH_WEAPON, gun_category]])
            tmp.append([drop_slot, 1, ActionType.ACTION_SWITCH_WEAPON, gun_category])
            # for att_ist_id in gun_att:
            #     # actions_seq.append([[att_ist_id, 1, ActionType.ACTION_DROP_ITEM, state['attachments'][att_ist_id][0]]])
            #     tmp.append([att_ist_id, 1, ActionType.ACTION_DROP_ITEM, state['attachments'][att_ist_id][0]])
        pick_idx = env_state[SupplyType.GUN][pick_idx][1]
        distance, num, x, y, z, category, obj_id, ist_id = env_supplys[pick_idx]
        # actions_seq.append([[int(obj_id), int(ist_id), ActionType.ACTION_PICK, category]])
        tmp.append([int(obj_id), int(ist_id), ActionType.ACTION_PICK, category])
        actions_seq.append(tmp)
    return change_gun, actions_seq

def plan_bag(env_supplys, env_state, state):
    actions_seq = []
    if state[SupplyType.BAG] < 0:
        change_bag = len(env_state[SupplyType.BAG]) > 0
    elif len(env_state[SupplyType.BAG]) > 0:
        change_bag = greater(env_state[SupplyType.BAG][0][0], state[SupplyType.BAG])
    else:
        change_bag = False

    if change_bag:
        pick_idx = env_state[SupplyType.BAG][0][1]
        distance, num, x, y, z, category, obj_id, ist_id = env_supplys[pick_idx]
        actions_seq.append([int(obj_id), int(ist_id), ActionType.ACTION_PICK, category])
    return change_bag, actions_seq

def plan_single_item(env_supplys, env_state, state):
    actions_seq = []
    for maintype in [SupplyType.CLOTHES, SupplyType.HELMET, SupplyType.SHIELD]:
        if state[maintype] < 0:
            change_item = len(env_state[maintype]) > 0
        elif len(env_state[maintype]) > 0:
            change_item = greater(env_state[maintype][0][0], state[maintype])
        else:
            change_item = False

        if change_item:
            pick_idx = env_state[maintype][0][1]
            distance, num, x, y, z, category, obj_id, ist_id = env_supplys[pick_idx]
            actions_seq.append([int(obj_id), int(ist_id), ActionType.ACTION_PICK, category])
    return actions_seq

def plan_attachments(env_supplys, env_state, state):
    att_states = []
    actions_seq = []
    for idx, my_gun in enumerate(state[SupplyType.GUN]):
        gun_category, gun_id, gun_atts, gun_inuse, gun_slot = my_gun
        att_state = defaultdict(lambda: [])
        for ist_id in gun_atts:
            gun_att = state['attachments'][ist_id]
            a_category, aid, a_cnt, inuse, a_maintype, a_subtype = gun_att
            a_subtype = AttachmentSubtype(a_subtype)
            att_state[a_subtype].append([a_category, ist_id])
        
        for subtype in attachments_non_special:
            for j, env_att in enumerate(env_state[subtype]):
                env_category, env_idx = env_att
                if env_category in all_supply_items[gun_category].attachment:
                    if len(att_state[subtype]) == 0:
                        pass
                    elif greater(env_category, state[subtype][0][0]):
                        item_cate, item_id = att_state[subtype][-1]
                        actions_seq.append([item_id, 1, ActionType.ACTION_DROP_ITEM, item_cate])
                    else:
                        continue
                    dist, count, x, y, z, category, obj_id, ist_id = env_supplys[env_idx]
                    actions_seq.append([int(obj_id), int(ist_id), ActionType.ACTION_PICK, category])
                    att_state[subtype].append(env_state[subtype].pop(j))
                    break
        att_states.append(att_state)
        
    all_scope_categories = []
    all_scopes = []
    for scope in state[AttachmentSubtype.SCOPE]:
        all_scopes.append(scope + [1])
        all_scope_categories.append(scope[0])
    for att in env_state[AttachmentSubtype.SCOPE]:
        if att[0] in all_scope_categories: continue
        scope = att + [0]
        all_scopes.append(scope)
    all_scopes = sort_supplys(all_scopes, 0)
    gun_count = len(state[SupplyType.GUN])

    scope_user = [-1 for _ in all_scopes]
    gun_choosed = [0 for _ in state[SupplyType.GUN]]

    for i in range(len(all_scopes) - 1, -1, -1):
        scope_category, scope_id, _ = all_scopes[i]
        for j, my_gun in enumerate(state[SupplyType.GUN]):
            gun_category, gun_id, gun_atts, gun_inuse, gun_slot = my_gun
            if scope_category in all_supply_items[gun_category].attachment:
                scope_user[i] = j
                gun_choosed[j] = 1
                break
        if sum(gun_choosed) > 0:
            break

    for i in range(len(all_scopes)):
        scope_category, scope_id, _ = all_scopes[i]
        for j, my_gun in enumerate(state[SupplyType.GUN]):
            if gun_choosed[j] > 0: continue
            gun_category, gun_id, gun_atts, gun_inuse, gun_slot = my_gun
            if scope_category in all_supply_items[gun_category].attachment:
                scope_user[i] = j
                gun_choosed[j] = 1
                break
        if sum(gun_choosed) > 1:
            break

    for i in range(len(all_scopes)):
        scope_category, scope_id, inbag = all_scopes[i]
        used = scope_user[i] >= 0
        if used:
            if inbag:
                dest_gun_scope = att_states[scope_user[i]][AttachmentSubtype.SCOPE]
                if len(dest_gun_scope) == 0:
                    pass # already equiped in previous stage
                elif scope_id == dest_gun_scope[0][-1]:
                    pass # already equiped before
                else:
                    actions_seq.append([scope_id, (scope_user[i] + 1) % 2 + 1, ActionType.ACTION_ATTACH, scope_category, False])
                    actions_seq.append([scope_id, scope_user[i] + 1, ActionType.ACTION_ATTACH, scope_category, True])
            else:
                dist, count, x, y, z, category, obj_id, ist_id = env_supplys[scope_id]
                actions_seq.append([int(obj_id), int(ist_id), ActionType.ACTION_PICK, category])
        else:
            if inbag:
                actions_seq.append([scope_id, 1, ActionType.ACTION_DROP_ITEM, scope_category])
            else:
                pass
    return actions_seq

def drop_bullet_attachment(state, player_supplys):
    actions_seq = []
    useful_bullet = []
    for i, gun in enumerate(state[SupplyType.GUN]):
        useful_bullet.append(all_supply_items[gun[0]].bullet)

    for i in player_supplys:
        if i[0] not in useful_bullet:
            actions_seq.append([i[1], i[2], ActionType.ACTION_DROP_ITEM, i[0]])

    for key, att_item in state['attachments'].items():
        p_cate, pid, p_cnt, inuse, p_main_type, p_subtype = att_item
        if not inuse:
            for i, gun in enumerate(state[SupplyType.GUN]):
                gun_cate, gun_id, gun_att, gun_inuse, gun_slot = gun
                if p_cate not in all_supply_items[gun_cate].attachment:
                    actions_seq.append([pid, p_cnt, ActionType.ACTION_DROP_ITEM, p_cate])

    return actions_seq

def plan_bullet(env_supplys, env_state, state):
    actions_seq = []
    max_num_idx = bag_level[state[SupplyType.BAG]]
    useful_bullet = set()
    for i, gun in enumerate(state[SupplyType.GUN]):
        gun_cate, gun_id, gun_atts, gun_inuse, gun_slot = gun
        useful_bullet.add(all_supply_items[gun[0]].bullet)

    for bullet in useful_bullet:
        total_num = state[bullet]
        for supply in env_state[bullet]:
            category, idx, count = supply
            if total_num + count < all_supply_items[category].max_num[max_num_idx]:
                distance, num, x, y, z, category, obj_id, ist_id = env_supplys[idx]
                actions_seq.append([int(obj_id), int(ist_id), ActionType.ACTION_PICK, category])
                total_num += num
            else:
                break
    return actions_seq

def plan_multiple_item(env_supplys, env_state, state):
    actions_seq = []
    max_num_idx = bag_level[state[SupplyType.BAG]]
    multiple_categories = [3301, 3302, 3401]

    for supply_category in multiple_categories:
        total_num = state[supply_category]
        for supply in env_state[supply_category]:
            category, idx, count = supply
            if total_num + count < all_supply_items[category].max_num[max_num_idx]:
                distance, num, x, y, z, category, obj_id, ist_id = env_supplys[idx]
                actions_seq.append([int(obj_id), int(ist_id), ActionType.ACTION_PICK, category])
                total_num += num
            else:
                break
    return actions_seq

def split_supplys(env_supplys, state):
    env_state = defaultdict(lambda: [])

    for idx, supply in enumerate(env_supplys):
        dist, count, x, y, z, category, obj_id, ist_id = supply
        main_type = all_supply_items[category].type
        if main_type in items_has_priority:
            if main_type not in [SupplyType.ATTACHMENT, SupplyType.GUN]:
                if len(env_state[main_type]) == 0:
                    env_state[main_type].append([category, idx])
                else:
                    curr = env_state[main_type][0]
                    env_state[main_type][0] = curr if not less(curr[0], category) else [category, idx]
            elif main_type == SupplyType.GUN:
                if len(env_state[main_type]) == 0:
                    env_state[main_type].append([category, idx])
                else:
                    snipe_cnt = 0
                    for gun in env_state[main_type]:
                        snipe_cnt += gun[0] in [2201, 2202, 2203]
                    if category in [2201, 2202, 2203]:
                        if snipe_cnt > 0:
                            curr = env_state[main_type][-1]
                            env_state[main_type][-1] = curr if not less(curr[0], category) else [category, idx]
                        else:
                            env_state[main_type].append([category, idx])
                    else:
                        if snipe_cnt > 0 and len(env_state[main_type]) == 1:
                            env_state[main_type].append([category, idx])
                        else:
                            curr = env_state[main_type][0]
                            env_state[main_type][0] = curr if not less(curr[0], category) else [category, idx]
                env_state[main_type] = sort_supplys(env_state[main_type], 0)
            else:
                subtype = AttachmentSubtype(all_supply_items[category].subtype)
                if subtype in attachments_special:
                    if len(env_state[subtype]) < 2:
                        env_state[subtype].append([category, idx])
                    else:
                        curr = env_state[subtype][0]
                        env_state[subtype][0] = curr if not greater(curr[0], category) else [category, idx]
                        env_state[subtype][1] = curr if not less(curr[0], category) else [category, idx]
                else:
                    if len(env_state[subtype]) < 2:
                        env_state[subtype].append([category, idx])
                    else:
                        curr = env_state[subtype][0]
                        env_state[subtype][0] = curr if not less(curr[0], category) else [category, idx]
                env_state[subtype] = sort_supplys(env_state[subtype], 0)
        else:
            env_state[category].append([category, idx, count])
    return env_state


def sort_supplys(supplys: list, category_idx):
    def sort_key(item):
        return all_supply_items[item[category_idx]].priority
    supplys.sort(key=sort_key)
    return supplys

def greater(category_0, category_1):
    return all_supply_items[category_0].priority > all_supply_items[category_1].priority

def less(category_0, category_1):
    return all_supply_items[category_0].priority < all_supply_items[category_1].priority

def diff(category_0, category_1):
    return all_supply_items[category_0].priority - all_supply_items[category_1].priority


if __name__ == '__main__':
    env_supplys = [[60.87608983632866, 1, 24146.48046875, 209911.640625, 2350.0, 3203, 175986, 0], [117.56498855256717, 1, 24138.125, 209843.921875, 2350.0, 200003001, 175694, 0], [60.87608983632866, 1, 24146.48046875, 209911.640625, 2350.0, 2403, 175061, 0], [117.56498855256717, 1, 24138.125, 209843.921875, 2350.0, 2403, 175765, 0], [60.87608983632866, 1, 24146.48046875, 209911.640625, 2350.0, 3212, 176281, 0], [117.56498855256717, 1, 24138.125, 209843.921875, 2350.0, 200002002, 174808, 0], [60.87608983632866, 1, 24146.48046875, 209911.640625, 2350.0, 200002004, 174896, 0], [60.87608983632866, 1, 24146.48046875, 209911.640625, 2350.0, 200002001, 174873, 0], [107.82115163504895, 1, 24188.4453125, 209847.96875, 2350.0, 200002001, 176245, 0], [117.56498855256717, 1, 24138.125, 209843.921875, 2350.0, 200004003, 175382, 0], [60.87608983632866, 1, 24146.48046875, 209911.640625, 2350.0, 2501, 175163, 0], [60.87608983632866, 1, 24146.48046875, 209911.640625, 2350.0, 3302, 175706, 0], [107.82115163504895, 1, 24188.4453125, 209847.96875, 2350.0, 3302, 175774, 0], [117.56498855256717, 1, 24138.125, 209843.921875, 2350.0, 3223, 175142, 0], [107.82115163504895, 1, 24188.4453125, 209847.96875, 2350.0, 2202, 175827, 0]]
    player_supplys = [
        [2201, 2, 1, True, SupplyType.GUN, 1],
        [2102, 1, 1, True, SupplyType.GUN, 1],
        [3222, 5, 1, True, SupplyType.BAG, 0],
        [3104, 10, 10, False, SupplyType.BULLET, 0],
        [3103, 9, 10, False, SupplyType.BULLET, 0],
        [3102, 8, 10, False, SupplyType.BULLET, 0],
        [3101, 7, 10, False, SupplyType.BULLET, 0],
        [3203, 3, 1, True, SupplyType.SHIELD, 0]
    ]
    player_state = {SupplyType.GUN: [[2103, 669, [], True, 1]], 3101: 0, 3102: 0, 3103: 0, 3104: 0, 3105: 0, 3401: 0, 3402: 0, 3403: 0, 3301: 0, 3302: 0, 3303: 0, SupplyType.CLOTHES: -1, SupplyType.HELMET: -1, SupplyType.SHIELD: -1, SupplyType.BAG: 3223, AttachmentSubtype.SILENCER: [], AttachmentSubtype.SCOPE: [], AttachmentSubtype.HANDLE: [], AttachmentSubtype.MAGAZINE: [], AttachmentSubtype.STOCK: [], 'attachments': {}}
    print(plan(env_supplys, player_supplys, player_state))
