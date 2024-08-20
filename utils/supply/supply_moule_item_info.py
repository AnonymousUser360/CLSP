import os
import xml
from xml.dom.minidom import parse
from collections import defaultdict
from .supply_type_info import SupplyType, AttachmentSubtype


def check_type(value, revalue=0, debug=False, method=int):
    if value is None or value == "":
        if debug:
            print(f'value is {value},i.e. None')
        return revalue
    return method(value)


need_space_types = [SupplyType.MEDICINE, SupplyType.BOMB]
DEPENDENT_TYPES = [SupplyType.ATTACHMENT, SupplyType.BULLET]
bag_size = defaultdict(lambda: 100)
bag_size.update({3221: 150, 3222: 200, 3223: 250})
bag_level = defaultdict(lambda: 0)
bag_level.update({3221: 1, 3222: 2, 3223: 3})
bag_categorys = [-1, 3221, 3222, 3223]


class SupplyItem:
    item_arr = ["id", "name",  "type", "subtype", 'size', 'max_num', 'bullet', 'attachment','priority', 'display', 'dependent']

    def __init__(self):
        for item in self.item_arr:
            self.__dict__[item] = None

    def update(self, item):
        self.id = check_type(item.getAttribute("id"))
        self.sub_id = self.id % 10
        self.name = item.getAttribute("name")
        self.display = item.getAttribute("display")
        self.type = SupplyType(check_type(item.getAttribute("type")))
        self.subtype = check_type(item.getAttribute("subtype",))
        self.size = check_type(item.getAttribute("size"), method=float)
        self.max_num = check_type(item.getAttribute("max_num"), method=eval)
        self.bullet = check_type(item.getAttribute("bullet",))
        self.priority = check_type(item.getAttribute("priority",))
        self.attachment = item.getAttribute('attachment')
        self.dependent = SupplyType(self.type) in DEPENDENT_TYPES
        if self.attachment:
            self.attachment = set(eval(self.attachment))

    def __repr__(self):
        return f"{self.id}-{self.name}"


class AllSupplyItems:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.dict_items = {}
        self.get_all_items()

    def get_all_items(self):
        DOMTree_supply_item = xml.dom.minidom.parse(self.xml_path)
        collection_supply_item = DOMTree_supply_item.documentElement
        for item_ in collection_supply_item.getElementsByTagName("supply_item"):
            supply_item = SupplyItem()
            supply_item.update(item_)
            self.dict_items[supply_item.id] = supply_item

    def __len__(self):
        return len(self.dict_items)

    def __getitem__(self, item):
        return self.dict_items[item]

    def keys(self):
        return self.dict_items.keys()

    def values(self):
        return self.dict_items.values()

    def items(self):
        return self.dict_items.items()


class VehicleItem:
    item_arr = ["id", "name",  "type", 'size', 'max_num', 'priority', 'display']

    def __init__(self):
        for item in self.item_arr:
            self.__dict__[item] = None

    def update(self, item):
        self.id = check_type(item.getAttribute("id"))
        self.name = item.getAttribute("name")
        self.display = item.getAttribute("display")
        self.type = SupplyType(check_type(item.getAttribute("type")))
        self.size = check_type(item.getAttribute("size"), method=float)
        self.max_num = check_type(item.getAttribute("max_num"), method=eval)
        self.priority = check_type(item.getAttribute("priority",))

    def __repr__(self):
        return f"{self.id}-{self.name}"


class AllVehicleItems:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.dict_items = {}
        self.get_all_items()

    def get_all_items(self):
        DOMTree_vehicle_item = xml.dom.minidom.parse(self.xml_path)
        collection_vehicle_item = DOMTree_vehicle_item.documentElement
        for item_ in collection_vehicle_item.getElementsByTagName("vehicle_item"):
            vehicle_item = VehicleItem()
            vehicle_item.update(item_)
            self.dict_items[vehicle_item.id] = vehicle_item

    def __len__(self):
        return len(self.dict_items)

    def __getitem__(self, item):
        return self.dict_items[item]

    def keys(self):
        return self.dict_items.keys()

    def values(self):
        return self.dict_items.values()

    def items(self):
        return self.dict_items.items()


all_supply_items = AllSupplyItems(xml_path=os.path.join(os.path.dirname(__file__), 'supply_module_item_info.xml'))
all_vehicle_items = AllSupplyItems(xml_path=os.path.join(os.path.dirname(__file__), 'supply_module_vehicle_info.xml'))

need_space_items = []
for k, item in all_supply_items.items():
    if item.type in need_space_types:
        need_space_items.append(item)

for k, item in all_supply_items.items():
    if item.max_num[0] < 0:
        for i, bcate in enumerate(bag_categorys):
            free_space = bag_size[bcate] - 20
            for sitem in need_space_items:
                free_space -= sitem.max_num[i] * sitem.size
            item.max_num[i] = int(free_space / item.size / 2)

AttachmentCategorys = []
GunCategorys = []
for k, item in all_supply_items.items():
    if item.type.value == 8:
        AttachmentCategorys.append(k)
    elif item.type.value == 1:
        GunCategorys.append(k)

AttachmentUseble = [[0 for _ in range(len(AttachmentCategorys))] for _ in range(len(GunCategorys) )] 
for i in range(len(GunCategorys)):
    for j in range(len(AttachmentCategorys)):
        if AttachmentCategorys[j] in all_supply_items[GunCategorys[i]].attachment:
            AttachmentUseble[i][j] = 1

from collections import defaultdict

gun2bullet= {}
bullet2gun = defaultdict(set)
for k, val in all_supply_items.items():
    if val.type == SupplyType.GUN:
        bullet_id = val.bullet
        bullet2gun[bullet_id].add(val.id)
        gun2bullet[val.id] = bullet_id

AttachmentsLocs = {
    "Muzzle":[200001001,200001002], 
    "grip":[200003001,200003002,], 
    "butt":[200005001,200005002,200005003], 
    "clip":[200004001,200004002,200004003], 
    "sight":[200002001,200002002,200002003,200002004,200002005,200002006]
}
RecoveryHP = [3301,3301]



class MONSTER:
    def __init__(self):
        self._data = {
            
            7001:1,
            7002:2,
            7003 :3,
            7004 :4,
            7005 :5,

            7301 :6,
            7302 :7,
            7303 :8,
            7304 :9,

            7701 :10,
            7702 :11,
        }

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()




if __name__ == '__main__':
    print(len(all_supply_items))
    print(all_supply_items.values())
    print(bullet2gun)
    print(gun2bullet)
    d={SupplyType.GUN: 100, AttachmentSubtype.SILENCER: 233}
    d[SupplyType(1)] = 1
    print(all_supply_items[3102].max_num)
    print(all_supply_items[3103].max_num)
    print([_ for _ in AttachmentSubtype])
