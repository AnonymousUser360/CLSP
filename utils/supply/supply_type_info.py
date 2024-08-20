# -*- coding:utf-8 -*
from enum import Enum, unique


@unique
class SupplyType(Enum):
    UNKNOWN = 0
    GUN = 1
    BULLET = 2
    BOMB = 3
    SHIELD = 4
    CLOTHES = 5
    HELMET = 6
    BAG = 7
    ATTACHMENT = 8
    MEDICINE = 9
    VEHICLE = 10


@unique
class WeaponSubtype(Enum):
    UNKNOWN = 0
    RIFLE = 1 # 突击步枪
    SNIPER = 2  # 狙击枪
    SUBMACHINE = 4  # 冲锋枪
    SHOTGUN = 5 # 霰弹枪
    MACHINE = 6 # 机枪
    SPECIAL = 8 # 特殊
    KNIFE = 10 # 近战

@unique
class AttachmentSubtype(Enum):
    UNKNOWN = 0
    SILENCER = 1    # 消音
    SCOPE = 2       # 倍镜
    HANDLE = 3      # 握把
    MAGAZINE = 4    # 弹匣
    STOCK = 5       # 枪托


if __name__ == '__main__':
    print(SupplyType(0).name, SupplyType(0).value)
    print(SupplyType['GUN'].name, SupplyType['GUN'].value)
