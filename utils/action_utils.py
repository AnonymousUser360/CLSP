# -*- coding:utf-8 -*


from fps.proto.ccs_ai import Action, AIAction, AIStateResponse, PlayerResultInfo, Vector2, Vector3, \
    ActionFocus, ActionMove, ActionRun, ActionSlide, ActionCrouch, ActionDrive, ActionJump, \
    ActionGround, ActionFire, ActionAim, ActionSwitchWeapon, ActionReload, ActionPick, ActionConsumeItem, \
    ActionDropItem, ActionHoldThrownItem, \
    ActionCancelThrow, ActionThrow, ActionRescue, ActionDoorSwitch, ActionSwim, ActionOpenAirDrop, ActionType, \
    ActionSkill, ActionTargetMove, ActionAttach

def target_move(pos, pid):
    # print("move to: ", pos)
    act = ActionTargetMove()
    act.ai_id = pid
    act.target_position = Vector3(pos[0], pos[1], pos[2])
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_TARGET_MOVE, action_data=am_bytes)
    return act


def move(pos, pid):
    act = ActionMove()
    act.ai_id = pid
    act.direction = Vector2(pos[0], pos[1])
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_MOVE, action_data=am_bytes)
    return act


def pick(obj_id, ist_id, pid):
    act = ActionPick(pid, ist_id, obj_id)
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_PICK, action_data=am_bytes)
    return act

def jump(pid):
    act = ActionJump(pid)
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_JUMP, action_data=am_bytes)
    return act


def drop(ist_id, cnt, pid):
    act = ActionDropItem(pid, ist_id, cnt)
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_DROP_ITEM, action_data=am_bytes)
    return act


def switch_door(pid,op,item_id):
    act = ActionDoorSwitch(pid, op, item_id)
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_DOOR_SWITCH, action_data=am_bytes)
    return act
def rescue_teammate(pid, tid):
    act = ActionRescue(pid, tid)
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_RESCUE, action_data=am_bytes)
    return act

def install_attachment(ist_id, slot_id, attach, pid):
    act = ActionAttach(pid, ist_id, slot_id, attach)
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_ATTACH, action_data=am_bytes)
    return act


def switch_weapon(slot_id, pid):
    act = ActionSwitchWeapon(pid, slot_id)
    am_bytes = bytes(act)
    act = Action(type=ActionType.ACTION_SWITCH_WEAPON, action_data=am_bytes)
    return act


def action_response(action_list, _id):
    rsp = AIStateResponse()
    pr = PlayerResultInfo()
    pr.id = _id
    pr.ai_action = AIAction()
    rsp.result = [pr]
    rsp.result[0].ai_action.actions = action_list
    return rsp

