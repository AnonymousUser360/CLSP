import math
import random
import numpy as np
from .supply_moule_item_info import all_supply_items as ALL_SUPPLY_ITEMS


def compute_dis(list1, list2):
    return math.sqrt((list2[0] - list1[0]) ** 2 + (list2[1] - list1[1]) ** 2 + (list2[2] - list1[2]) ** 2)


def compute_xyz2scalar(list1):
    return math.sqrt(list1[0] ** 2 + list1[1] ** 2 + list1[2] ** 2)


def distance_via_id(id_1, id_2, info=None):
    x1 = info['player_state'][id_1].state.position.x
    y1 = info['player_state'][id_1].state.position.y
    z1 = info['player_state'][id_1].state.position.z
    x2 = info['player_state'][id_2].state.position.x
    y2 = info['player_state'][id_2].state.position.y
    z2 = info['player_state'][id_2].state.position.z
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def choice_enemy(state_info_queue, player_id):
    team_id = state_info_queue[0][1]['player_state'][player_id].state.team_id

    nearst_enemy_dis = np.inf
    nearst_enemy_id = None

    for pid, state_info in state_info_queue[-1][1]['player_state'].items():
        if pid != player_id and team_id != state_info.state.team_id:
            pid_dis = distance_via_id(player_id, pid, state_info_queue[0][1])
            if pid_dis < nearst_enemy_dis:
                nearst_enemy_dis = pid_dis
                nearst_enemy_id = pid
    return nearst_enemy_id


def init_target_id(state_info, player_id):
    all_teammate = []

    team_id = state_info[-1][1]['player_state'][player_id].state.team_id

    for pid, v in state_info[-1][1]['player_state'].items():
        if v.state.team_id == team_id and pid != player_id:
            all_teammate.append(pid)

    all_teammate.append(player_id)
    all_teammate = sorted(all_teammate)
    init_target_index = all_teammate.index(player_id)

    return init_target_index


def choice_teammate_follow(state_info, teammate_index):
    all_teammate = []
    alive_teammate = []

    player_id = state_info[0][2]
    team_id = state_info[0][1]['player_state'][player_id].state.team_id

    for pid, v in state_info[-1][1]['player_state'].items():
        if v.state.team_id == team_id and pid != player_id and v.state.alive_state != 2:
            alive_teammate.append(pid)
        if v.state.team_id == team_id and pid != player_id:
            all_teammate.append(pid)

    all_teammate.append(player_id)
    all_teammate = sorted(all_teammate)
    if len(all_teammate) < teammate_index or teammate_index == 0:
        return None
    else:
        return all_teammate[teammate_index - 1]


def if_visable_player(team_id, player_id, visable, info, check_enemy=True):
    visable_players = []
    if len(visable) == 0:
        return []
    else:
        for pid, v in info['player_state'].items():
            if check_enemy:
                if pid != player_id and v.state.team_id != team_id and (pid in visable):
                    visable_players.append(pid)
            else:
                if pid != player_id and v.state.team_id == team_id and (pid in visable):
                    visable_players.append(pid)

    return visable_players


def calc_enemy_num(team_id, state_info):
    num_enemy = 0
    for pid, v in state_info[1]['player_state'].items():
        if v.state.team_id != team_id:
            num_enemy += 1
    return num_enemy


def teammate_id_compu(state_info_queue, target_id):
    if target_id is None:
        return 0
    player_id = state_info_queue[0][2]
    team_id = state_info_queue[0][1]['player_state'][player_id].state.team_id

    alive_teammate = []
    alive_teamate_dis_list = []
    for pid, v in state_info_queue[-1][1]['player_state'].items():
        if v.state.team_id == team_id and pid != player_id and v.state.alive_state != 2:
            alive_teammate.append(pid)
            alive_teamate_dis = distance_via_id(pid, player_id, state_info_queue[-1][1])
            alive_teamate_dis_list.append(alive_teamate_dis)

    if target_id not in alive_teammate:
        return 0
    nearest_teammate_dis_morm = 0
    alive_teamate_dis = distance_via_id(player_id, target_id, state_info_queue[-1][1])
    if alive_teamate_dis < 1000:
        nearest_teammate_dis_morm = 1
    elif alive_teamate_dis < 2000:
        nearest_teammate_dis_morm = 2
    elif alive_teamate_dis < 5000:
        nearest_teammate_dis_morm = 3
    elif alive_teamate_dis < 15000:
        nearest_teammate_dis_morm = 4
    else:
        nearest_teammate_dis_morm = 5

    return nearest_teammate_dis_morm


def damage_kill_knockdown(state_info_queue, player_id,cusum_meta = {"damage":0,
                                                                        "kill":0,
                                                                        "knockdown":0,}):
    if player_id is None:
        return 0, 0, 0

    player_damage = state_info_queue[-1][1]['player_state'][player_id].statistic.damage
    if len(state_info_queue) > 1:
        last_player_damage = state_info_queue[-2][1]['player_state'][player_id].statistic.damage
    else:
        last_player_damage = player_damage
    delta_damage = player_damage - last_player_damage
    damage_norm = 0
    if delta_damage <= 0:
        damage_norm = 0
    elif delta_damage < 20:
        damage_norm = 1
    elif delta_damage < 40:
        damage_norm = 2
    elif delta_damage < 60:
        damage_norm = 3
    elif delta_damage < 80:
        damage_norm = 4
    else:
        damage_norm = 5

    player_knock_done = state_info_queue[-1][1]['player_state'][player_id].statistic.knock_done
    if len(state_info_queue) > 1:
        last_player_knock_done = state_info_queue[-2][1]['player_state'][player_id].statistic.knock_done
    else:
        last_player_knock_done = player_knock_done
    delta_knock_done = (player_knock_done - last_player_knock_done) > 0

    player_kill = state_info_queue[-1][1]['player_state'][player_id].statistic.kill
    if len(state_info_queue) > 1:
        last_player_kill = state_info_queue[-2][1]['player_state'][player_id].statistic.kill
    else:
        last_player_kill = player_kill
    delta_kill = (player_kill - last_player_kill) > 0

    cusum_meta["damage"] = min(5,damage_norm + cusum_meta["damage"])
    cusum_meta["kill"] = (delta_kill + cusum_meta["kill"]) > 0
    cusum_meta["knockdown"] = (delta_knock_done + cusum_meta["knockdown"]) > 0

    return cusum_meta["damage"], cusum_meta["knockdown"], cusum_meta["kill"]


def nearst_enemy_dis_from_goal_start(state_info_queue, player_id):
    if player_id is None:
        return 0, 0

    team_id = state_info_queue[0][1]['player_state'][player_id].state.team_id

    nearst_enemy_dis = np.inf
    nearst_enemy_id = None

    for pid, state_info in state_info_queue[-1][1]['player_state'].items():
        if pid != player_id and team_id != state_info.state.team_id:
            pid_dis = distance_via_id(player_id, pid, state_info_queue[0][1])
            if pid_dis < nearst_enemy_dis:
                nearst_enemy_dis = pid_dis
                nearst_enemy_id = pid

    if len(state_info_queue) < 2:
        distance_delta =  distance_via_id(player_id, nearst_enemy_id, state_info_queue[-1][1]) - distance_via_id(
            player_id, nearst_enemy_id, state_info_queue[-1][1]) 
    else:
        distance_delta =  distance_via_id(player_id, nearst_enemy_id, state_info_queue[-1][1]) - distance_via_id(
            player_id, nearst_enemy_id, state_info_queue[-2][1]) 
    
    if distance_delta>0:
        two_step_dis_dif = True
    else:
        two_step_dis_dif = False



    nearst_enemy_dis_norm = 0
    if nearst_enemy_dis < 1000:
        nearst_enemy_dis_norm = 0
    elif nearst_enemy_dis < 5000:
        nearst_enemy_dis_norm = 1
    elif nearst_enemy_dis < 10000:
        nearst_enemy_dis_norm = 2
    elif nearst_enemy_dis < 20000:
        nearst_enemy_dis_norm = 3
    elif nearst_enemy_dis < 50000:
        nearst_enemy_dis_norm = 4
    else:
        nearst_enemy_dis_norm = 5

    return nearst_enemy_dis_norm, two_step_dis_dif


def cal_nearest_enemy_info(state_info_queue,goal_start_state,if_goal_mode, player_id):
    team_id = state_info_queue[0][1]['player_state'][player_id].state.team_id

    nearst_enemy_dis = np.inf
    nearst_enemy_id = None

    for pid, state_info in state_info_queue[-1][1]['player_state'].items():
        if pid != player_id and team_id != state_info.state.team_id:
            pid_dis = distance_via_id(player_id, pid, state_info_queue[0][1])
            if pid_dis < nearst_enemy_dis:
                nearst_enemy_dis = pid_dis
                nearst_enemy_id = pid

    enemy_move_dir_norm = move_dir(state_info_queue, nearst_enemy_id)
    enemy_delta_pos, enemy_mean_speed = calc_delta_pos(state_info_queue, goal_start_state,if_goal_mode = if_goal_mode,player_id = nearst_enemy_id)
    enemy_yaw_state, enemy_pitch_state = process_yaw_pitch(state_info_queue, nearst_enemy_id)

    return enemy_move_dir_norm, enemy_mean_speed, enemy_yaw_state


def whether_see_enemy(state_info_queue, player_id):
    if player_id is None:
        return 0, 0, 0

    list_see_enemy_raw = []

    team_id = state_info_queue[0][1]['player_state'][player_id].state.team_id

    see_enemy_in10step = set()

    for idx, state_info in enumerate(list(state_info_queue)):
        enemy_num = calc_enemy_num(team_id, state_info)
        if enemy_num >= 1:
            visable_ids = state_info[1]['player_state'][player_id].visble_player_ids
            if len(visable_ids) == 0:
                see_enemy = False
            else:
                visable_players = if_visable_player(team_id, player_id, visable_ids, state_info[1], check_enemy=True)
                see_enemy_in10step.update(visable_players)
                if len(visable_players) > 0:
                    see_enemy = True
                else:
                    see_enemy = False
        elif enemy_num == 0:
            see_enemy = False
        else:
            see_enemy = False
        list_see_enemy_raw.append(see_enemy)

    list_see_enemy = sum(list_see_enemy_raw) > 0
    list_see_enemy_last_10 = list_see_enemy_raw[-1]
    see_enemy_in10step = list(see_enemy_in10step)

    return list_see_enemy, list_see_enemy_last_10, len(see_enemy_in10step)


def check_players_can_see_me(team_id, player_id, info, check_enemy=True):
    visable_players = []

    for pid, v in info['player_state'].items():
        visble_player_ids = info['player_state'][pid].visble_player_ids
        if check_enemy:
            if pid != player_id and v.state.team_id != team_id and (player_id in visble_player_ids):
                visable_players.append(pid)
        else:
            if pid != player_id and v.state.team_id == team_id and (player_id in visble_player_ids):
                visable_players.append(pid)

    return visable_players


def whether_enemy_can_seeme(state_info_queue, player_id):
    if player_id is None:
        return 0, 0

    list_enemy_see_raw = []
    team_id = state_info_queue[0][1]['player_state'][player_id].state.team_id

    for state_info in list(state_info_queue):
        if len(check_players_can_see_me(team_id, player_id, state_info[1], check_enemy=True)) > 0:
            enemy_can_seeme = True
        else:
            enemy_can_seeme = False
        list_enemy_see_raw.append(enemy_can_seeme)

    list_enemy_see = sum(list_enemy_see_raw) > 0
    list_enemy_see_last_10 = list_enemy_see_raw[-1]

    return list_enemy_see, list_enemy_see_last_10


def whether_teammate_can_seeme(state_info_queue, player_id):
    if player_id is None:
        return 0, 0

    list_teammate_see_raw = []
    team_id = state_info_queue[0][1]['player_state'][player_id].state.team_id

    for state_info in list(state_info_queue):
        if len(check_players_can_see_me(team_id, player_id, state_info[1], check_enemy=False)) > 0:
            teammate_can_seeme = True
        else:
            teammate_can_seeme = False
        list_teammate_see_raw.append(teammate_can_seeme)

    list_teammate_see = sum(list_teammate_see_raw) > 0
    list_teammate_see_last_10 = list_teammate_see_raw[-1]

    return list_teammate_see, list_teammate_see_last_10


def calc_delta_pos(state_info_queue,goal_start_state, player_id, if_goal_mode):
    if (player_id is None):
        return 0, 0 

    player_speed_scalar_list = []
    for state_info in list(state_info_queue)[-5:]:
        player_speed = state_info[1]['player_state'][player_id].state.speed
        player_speed_scalar = compute_xyz2scalar([player_speed.x, player_speed.y, player_speed.z])
        player_speed_scalar_list.append(player_speed_scalar)
    mean_speed = player_speed_scalar_list[-1]
    mean_speed_norm = 0
    if mean_speed < 1:
        mean_speed_norm = 0
    elif mean_speed < 200:
        mean_speed_norm = 1
    elif mean_speed < 500:
        mean_speed_norm = 2
    elif mean_speed < 800:
        mean_speed_norm = 3
    else:
        mean_speed_norm = 4

    if (not if_goal_mode): # 非goal模式的距离为0
        pos = state_info_queue[-1][1]['player_state'][player_id].state.position
        
        last_idx = 0 if len(state_info_queue) <= 50 else -50

        pos_last = state_info_queue[last_idx][1]['player_state'][player_id].state.position # 固定住开始的state

        pos_last = [pos_last.x, pos_last.y, pos_last.z]
        pos = [pos.x, pos.y, pos.z]
        delta_pos = compute_dis(pos, pos_last)
        delta_pos_norm = 0
        if delta_pos < 200:
            delta_pos_norm = 0
        elif delta_pos < 1500:
            delta_pos_norm = 1
        elif delta_pos < 3000:
            delta_pos_norm = 2
        elif delta_pos < 4500:
            delta_pos_norm = 3
        else:
            delta_pos_norm = 4
        return delta_pos_norm, mean_speed_norm
    
    else:
    
        pos = state_info_queue[-1][1]['player_state'][player_id].state.position

        pos_last = goal_start_state[1]['player_state'][player_id].state.position # 固定住开始的state

        pos_last = [pos_last.x, pos_last.y, pos_last.z]
        pos = [pos.x, pos.y, pos.z]
        delta_pos = compute_dis(pos, pos_last)
        delta_pos_norm = 0
        if delta_pos < 200:
            delta_pos_norm = 0
        elif delta_pos < 1500:
            delta_pos_norm = 1
        elif delta_pos < 3000:
            delta_pos_norm = 2
        elif delta_pos < 4500:
            delta_pos_norm = 3
        else:
            delta_pos_norm = 4

        return delta_pos_norm, mean_speed_norm


def relative_dir(state_info_queue, object_id, subject_id):
    if subject_id is None:
        return 0
    object_pos = state_info_queue[-1][1]['player_state'][object_id].state.position
    subject_pos = state_info_queue[-1][1]['player_state'][subject_id].state.position

    dx = subject_pos.x - object_pos.x
    dy = subject_pos.y - object_pos.y
    dir = np.arctan2(dy, dx) / np.pi * 180 + 180

    if dir >= 0 and dir < 45:
        move_dir_norm = 0
    elif dir >= 45 and dir < 90:
        move_dir_norm = 1
    elif dir >= 90 and dir < 135:
        move_dir_norm = 2
    elif dir >= 135 and dir < 180:
        move_dir_norm = 3
    elif dir >= 180 and dir < 225:
        move_dir_norm = 4
    elif dir >= 225 and dir < 270:
        move_dir_norm = 5
    elif dir >= 270 and dir < 315:
        move_dir_norm = 6
    elif dir >= 315 and dir < 360:
        move_dir_norm = 7
    else:
        move_dir_norm = 0

    return move_dir_norm


def move_dir(state_info_queue, player_id):
    if player_id is None:
        return 0
    if len(state_info_queue) > 1:
        last_pos = state_info_queue[-2][1]['player_state'][player_id].state.position
    else:
        last_pos = state_info_queue[-1][1]['player_state'][player_id].state.position
    
    pos = state_info_queue[-1][1]['player_state'][player_id].state.position
    dx = pos.x - last_pos.x
    dy = pos.y - last_pos.y
    yaw_mine = np.arctan2(dy, dx) / np.pi * 180 + 180

    move_dir_norm = 0
    delat = -23 # 防止小数精度
    if (yaw_mine >= 0 and yaw_mine < 45+delat) or (yaw_mine >= 360+delat):
        move_dir_norm = 0
    elif yaw_mine >= 45+delat and yaw_mine < 90+delat:
        move_dir_norm = 1
    elif yaw_mine >= 90+delat and yaw_mine < 135+delat:
        move_dir_norm = 2
    elif yaw_mine >= 135+delat and yaw_mine < 180+delat:
        move_dir_norm = 3
    elif yaw_mine >= 180+delat and yaw_mine < 225+delat:
        move_dir_norm = 4
    elif yaw_mine >= 225+delat and yaw_mine < 270+delat:
        move_dir_norm = 5
    elif yaw_mine >= 270+delat and yaw_mine < 315+delat:
        move_dir_norm = 6
    elif yaw_mine >= 315+delat and yaw_mine < 360+delat:
        move_dir_norm = 7
    else:
        move_dir_norm = 0

    return move_dir_norm


def process_see_same_enemy(state_info_queue, object_id, subject_id):
    if subject_id is None:
        return 0

    info = state_info_queue[-1][1]
    common_visbles = set(info['player_state'][object_id].visble_player_ids) & \
                     set(info['player_state'][subject_id].visble_player_ids)

    return len(common_visbles) > 0


def process_yaw_dir(state_info_queue, player_id):
    if player_id is None:
        return 0
    camera_rotation = state_info_queue[-1][1]['player_state'][player_id].camera.rotation

    yaw_camera = camera_rotation.z

    yaw_state = 0
    delta = 23
    if (yaw_camera >= 0 and yaw_camera < 45 - delta) or (yaw_camera >= 360- delta):
        yaw_state = 0
    elif yaw_camera >= 45- delta and yaw_camera < 90- delta:
        yaw_state = 1
    elif yaw_camera >= 90- delta and yaw_camera < 135- delta:
        yaw_state = 2
    elif yaw_camera >= 135- delta and yaw_camera < 180- delta:
        yaw_state = 3
    elif yaw_camera >= 180- delta and yaw_camera < 225- delta:
        yaw_state = 4
    elif yaw_camera >= 225- delta and yaw_camera < 270- delta:
        yaw_state = 5
    elif yaw_camera >= 270- delta and yaw_camera < 315- delta:
        yaw_state = 6
    elif yaw_camera >= 315- delta and yaw_camera < 360- delta:
        yaw_state = 7
    else:
        yaw_state = 0

    return yaw_state


def process_yaw_pitch(state_info_queue, player_id):
    if player_id is None:
        return 0, 0

    camera_rotation = state_info_queue[-1][1]['player_state'][player_id].camera.rotation

    # TODO 确定环境返回值
    yaw_camera = camera_rotation.z
    pitch_camera = camera_rotation.y
    pitch_camera = pitch_camera if pitch_camera <= 90 else pitch_camera - 360

    yaw_state = 0
    pitch_state = 0
    delta = 23
    if (yaw_camera >= 0 and yaw_camera < 45-delta) or (yaw_camera >= 360-delta):
        yaw_state = 0
    elif yaw_camera >= 45-delta and yaw_camera < 90-delta:
        yaw_state = 1
    elif yaw_camera >= 90-delta and yaw_camera < 135-delta:
        yaw_state = 2
    elif yaw_camera >= 135-delta and yaw_camera < 180-delta:
        yaw_state = 3
    elif yaw_camera >= 180-delta and yaw_camera < 225-delta:
        yaw_state = 4
    elif yaw_camera >= 225-delta and yaw_camera < 270-delta:
        yaw_state = 5
    elif yaw_camera >= 270-delta and yaw_camera < 315-delta:
        yaw_state = 6
    elif yaw_camera >= 315-delta and yaw_camera < 360-delta:
        yaw_state = 7
    else:
        yaw_state = 0

    if pitch_camera < -40:
        pitch_state = 0
    elif pitch_camera > 40:
        pitch_state = 2
    else:
        pitch_state = 1

    return yaw_state, pitch_state


# def process_yaw_pitch(state_info_queue, player_id):
#     if player_id is None:
#         return 0, 0


#     camera_rotation = state_info_queue[-1][1]['player_state'][player_id].camera.rotation

#     yaw_camera = camera_rotation.z
#     pitch_camera = camera_rotation.y

#     pitch_camera = pitch_camera if pitch_camera <= 90 else pitch_camera - 360


#     yaw_state = 0
#     pitch_state = 0
#     #TODO
#     yaw_camera_id = int((yaw_camera) / (360 / 24)) # 不能加180
#     ### fixme
#     if yaw_camera_id == 24:
#         yaw_camera_id = 23
#     # assert 0 <= yaw_camera_id <= 23
#     if yaw_camera_id == 23 or yaw_camera_id == 0 or yaw_camera_id == 1:
#         yaw_state = 0
#     elif yaw_camera_id == 2 or yaw_camera_id == 3 or yaw_camera_id == 4:
#         yaw_state = 1
#     elif yaw_camera_id == 5 or yaw_camera_id == 6 or yaw_camera_id == 7:
#         yaw_state = 2
#     elif yaw_camera_id == 8 or yaw_camera_id == 9 or yaw_camera_id == 10:
#         yaw_state = 3
#     elif yaw_camera_id == 11 or yaw_camera_id == 12 or yaw_camera_id == 13:
#         yaw_state = 4
#     elif yaw_camera_id == 14 or yaw_camera_id == 15 or yaw_camera_id == 16:
#         yaw_state = 5
#     elif yaw_camera_id == 17 or yaw_camera_id == 18 or yaw_camera_id == 19:
#         yaw_state = 6
#     elif yaw_camera_id == 20 or yaw_camera_id == 21 or yaw_camera_id == 22:
#         yaw_state = 7
#     else:
#         yaw_state = 0

#     pitch_camera_id = int((pitch_camera + 75) / (150 / 5))
#     if pitch_camera_id == 5:
#         pitch_camera_id = 4
#     ## fixme
#     # assert 0 <= pitch_camera_id <= 4
#     if pitch_camera_id == 0:
#         pitch_state = 0
#     elif pitch_camera_id == 1:
#         pitch_state = 1
#     elif pitch_camera_id == 2:
#         pitch_state = 2
#     elif pitch_camera_id == 3:
#         pitch_state = 3
#     elif pitch_camera_id == 4:
#         pitch_state = 4
#     else:
#         pitch_state = 0

#     print(f"yaw_camera is {yaw_camera}, yaw_camera_id is {yaw_camera_id}")
#     print(f"pitch_camera is {pitch_camera}, pitch_camera_id is {pitch_camera_id}")

#     return yaw_state, pitch_state
# TODO raise cheack
def health_level(state_info_queue, player_id):
    if player_id is None:
        return 0, 0, 0

    hp = state_info_queue[-1][1]['player_state'][player_id].state.hp
    hp_state = 0
    if hp < 1:
        hp_state = 0
    elif hp < 33:
        hp_state = 1
    elif hp < 66:
        hp_state = 2
    elif hp < 99:
        hp_state = 3
    else:
        hp_state = 4

    if len(state_info_queue) > 1:
        last_hp = state_info_queue[-2][1]['player_state'][player_id].state.hp
    else:
        last_hp = hp
    delta_hp = hp - last_hp
    if delta_hp == 0:
        hp_go_up = False
        hp_go_down = False
    elif delta_hp < -20:
        hp_go_up = False
        hp_go_down = True
    elif delta_hp > 20:
        hp_go_up = True
        hp_go_down = False
    else:
        hp_go_up = False
        hp_go_down = False

    return hp_state, hp_go_up, hp_go_down


def whether_rescue(state_info_queue, player_id):
    if player_id is None:
        return 0, 0

    list_rescue_raw = []
    list_be_rescued_raw = []

    for state_info in state_info_queue:
        rescue = state_info[1]['player_state'][player_id].progress_bar.type == 2
        list_rescue_raw.append(rescue)
        be_rescued = state_info[1]['player_state'][player_id].progress_bar.type == 3
        list_be_rescued_raw.append(be_rescued)

    list_rescue = sum(list_rescue_raw) > 0
    list_be_rescued = sum(list_rescue_raw) > 0
    return list_rescue, list_be_rescued


def whether_knocked_down(state_info_queue, player_id):
    if player_id is None:
        return 0

    alive_state = state_info_queue[-1][1]['player_state'][player_id].state.alive_state
    knocked_down = (alive_state == 1)
    return knocked_down


# def whether_be_knock_down(state_info_queue):
#     player_id = state_info_queue[0][2]
#     be_knock_down_times = state_info_queue[-1][1]['player_state'][player_id].statistic.be_knockdown_times
#     if len(state_info_queue) < 2:
#         last_be_knock_down_times = state_info_queue[-1][1]['player_state'][player_id].statistic.be_knockdown_times
#     else:
#         last_be_knock_down_times = state_info_queue[-2][1]['player_state'][player_id].statistic.be_knockdown_times
#     knock_down_times = (be_knock_down_times - last_be_knock_down_times) > 0
#     return knock_down_times

def whether_prone_position(state_info_queue, player_id):
    if player_id is None:
        return 0

    body_state = state_info_queue[-1][1]['player_state'][player_id].state.body_state
    prone_position = (body_state == 8)
    return prone_position


def whether_crouch_position(state_info_queue, player_id):
    if player_id is None:
        return 0

    body_state = state_info_queue[-1][1]['player_state'][player_id].state.body_state
    crouch_position = (body_state == 2)
    return crouch_position


#

def whether_gun_in_hand(state_info_queue, player_id):
    if player_id is None:
        return 0

    gun_in_hand_state = state_info_queue[-1][1]['player_state'][player_id].state.active_weapon_slot
    gun_in_hand_state = not (gun_in_hand_state == 0)
    return gun_in_hand_state


def whether_driving(state_info_queue):
    player_id = state_info_queue[0][2]
    body_state = state_info_queue[-1][1]['player_state'][player_id].state.body_state
    driving = body_state == 3
    return driving


def whether_have_gun(state_info_queue, player_id):
    if player_id is None:
        return 0

    guns = state_info_queue[-1][1]['player_state'][player_id].weapon.player_weapon
    have_gun = (len(guns) > 0)
    return have_gun


def whether_have_bullet(state_info_queue, player_id):
    if player_id is None:
        return 0

    guns = state_info_queue[-1][1]['player_state'][player_id].weapon.player_weapon
    if len(guns) == 0:
        bullets = 0
    elif len(guns) == 1:
        bullets = state_info_queue[-1][1]['player_state'][player_id].weapon.player_weapon[0].bullet
    elif len(guns) >= 2:
        bullets = state_info_queue[-1][1]['player_state'][player_id].weapon.player_weapon[0].bullet + \
                  state_info_queue[-1][1]['player_state'][player_id].weapon.player_weapon[1].bullet

    have_bullet = bullets > 0
    return have_bullet


def whether_have_medical_kits(state_info_queue, player_id):
    if player_id is None:
        return 0

    backpack_items = state_info_queue[-1][1]['player_state'][player_id].backpack.backpack_item

    medical_kits = 0
    if len(backpack_items) == 0:
        medical_kits = 0
    elif len(backpack_items) > 1:
        for item in backpack_items:
            backpack_item_class = ALL_SUPPLY_ITEMS[item.category]  # [3101,3102,3103]
            backpack_item_type = backpack_item_class.type.value  # 大类  1~10可限制为
            if backpack_item_type == 9:
                medical_kits += 1
    have_medical_kits = medical_kits > 0
    return have_medical_kits



def body_state_agger(state_info_queue,player_id):

    fire = state_info_queue[-1][1]['player_state'][player_id].state.is_firing 
    body_state = state_info_queue[-1][1]['player_state'][player_id].state.body_state 
    jump = body_state == 1
    swim = body_state == 6
    fall = state_info_queue[-1][1]['player_state'][player_id].state.is_falling 

    return {
        "fire":fire,
        "jump":jump,
        "swim":swim,
        "fall":fall,
    }



def get_height_z(state_info_queue, player_id):
    pos = state_info_queue[-1][1]['player_state'][player_id].state.position

    z = pos.z
    
    if z<5000:
        z_norm=0
    elif z<10000:
        z_norm=1
    elif z<15000:
        z_norm=2
    elif z<20000:
        z_norm=3
    else:
        z_norm=4

    return z_norm

def goto_door(state_info_queue, player_id, door_id):

    own_position = state_info_queue[-1][1]['player_state'][player_id].state.position
    door_position = state_info_queue[-1][1]['doors'][door_id].position
    dist2door = compute_dis([own_position.x,own_position.y,own_position.z],[door_position.x,door_position.y,door_position.z])

    if dist2door< 500:
        dist2door_norm = 0
    elif dist2door< 2000:
        dist2door_norm = 1
    elif dist2door< 5000:
        dist2door_norm = 2
    elif dist2door< 10000:
        dist2door_norm = 3
    else:
        dist2door_norm = 4
    return dist2door_norm

def get_open_close_door(state_info_queue, door_id):
    return state_info_queue[-1][1]['doors'][door_id].state > 0

def get_activate_weapon_id(state_info_queue, player_id):

    return state_info_queue[-1][1]['player_state'][player_id].state.active_weapon_slot

def check_in_circle(state_info_queue, player_id):
    player_pos = state_info_queue[-1][1]['player_state'][player_id].state.position
    circle_info = state_info_queue[-1][0].safety_area

    circle_radius = circle_info.radius
    next_circle_radius = circle_info.next_radius
    now_circle_pos = [circle_info.center.x, circle_info.center.y, 0]
    next_circle_pos = [circle_info.next_center.x, circle_info.next_center.y, 0]
    player_pos_list = [player_pos.x, player_pos.y, 0]

    in_blue_circle = (compute_dis(player_pos_list, now_circle_pos) < circle_radius)
    in_white_circle = (compute_dis(player_pos_list, next_circle_pos) < next_circle_radius)

    return in_blue_circle, in_white_circle

def check_8_dir_shelter(state_info_queue, player_id):
    ray_norm = 1000

    player_ray16 = state_info_queue[-1][1]['player_state'][player_id].dirs_ray_distance

    ray_dir0 = (player_ray16[0]/ray_norm < 0.2) or (player_ray16[1]/ray_norm < 0.2)
    ray_dir1 = (player_ray16[2]/ray_norm < 0.2) or (player_ray16[3]/ray_norm < 0.2)
    ray_dir2 = (player_ray16[4]/ray_norm < 0.2) or (player_ray16[5]/ray_norm < 0.2)
    ray_dir3 = (player_ray16[6]/ray_norm < 0.2) or (player_ray16[7]/ray_norm < 0.2)
    ray_dir4 = (player_ray16[8]/ray_norm < 0.2) or (player_ray16[9]/ray_norm < 0.2)
    ray_dir5 = (player_ray16[10]/ray_norm < 0.2) or (player_ray16[11]/ray_norm < 0.2)
    ray_dir6 = (player_ray16[12]/ray_norm < 0.2) or (player_ray16[13]/ray_norm < 0.2)
    ray_dir7 = (player_ray16[14]/ray_norm < 0.2) or (player_ray16[15]/ray_norm < 0.2)

    return ray_dir0, ray_dir1, ray_dir2, ray_dir3, ray_dir4, ray_dir5, ray_dir6, ray_dir7