import os
import xml
from xml.dom.minidom import parse


def check_int(value, revalue=0, debug=False):
    if value is None or value == "":
        if debug:
            print(f'value is {value},i.e. None')
        return revalue
    return int(value)


class VehicleItem:
    vehicle_item_arr = ["id", "name", "display_name", "type", "subtype"]

    def __init__(self):
        for item in self.vehicle_item_arr:
            self.__dict__[item] = None

    def update(self, item):
        self.id = check_int(item.getAttribute("id"))
        self.name = item.getAttribute("name")
        self.display_name = item.getAttribute("display_name")
        self.type = check_int(item.getAttribute("type"))
        self.subtype = check_int(item.getAttribute("subtype"))
        self.sub_id = self.id % 10

    def __repr__(self):
        return f"id:{self.id}, name:{self.display_name}, rep {self.type, self.subtype, self.sub_id}"


class AllVehicleItems:
    def __init__(self, xml_path='vehicle_item_info.xml'):
        self.xml_path = xml_path
        self.dict_vehicle_items = {}
        self.ignore_ids = {}
        self.get_all_vehicle_items()

    def get_all_vehicle_items(self):
        DOMTree_vehicle_item = xml.dom.minidom.parse(self.xml_path)
        collection_vehicle_item = DOMTree_vehicle_item.documentElement
        for item_ in collection_vehicle_item.getElementsByTagName("vehicle_item"):
            vehicle_item = VehicleItem()
            vehicle_item.update(item_)
            self.dict_vehicle_items[vehicle_item.id] = vehicle_item

    def __len__(self):
        return len(self.dict_vehicle_items)

    def __getitem__(self, item):
        return self.dict_vehicle_items[item]

    def keys(self):
        return self.dict_vehicle_items.keys()

    def values(self):
        return self.dict_vehicle_items.values()

    def items(self):
        return self.dict_vehicle_items.items()


all_vehicle_items = AllVehicleItems(xml_path=os.path.join(os.path.dirname(__file__), 'vehicle_item_info.xml'))

if __name__ == '__main__':
    print(len(all_vehicle_items))
    print(all_vehicle_items.values())
