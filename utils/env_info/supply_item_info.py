import os
import xml
from xml.dom.minidom import parse


def check_int(value, revalue=0, debug=False):
    if value is None or value == "":
        if debug:
            print(f'value is {value},i.e. None')
        return revalue
    return int(value)


class SupplyItem:
    supply_item_arr = ["id", "name", "display_name", "type", "subtype"]

    def __init__(self):
        for item in self.supply_item_arr:
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


class AllSupplyItems:
    def __init__(self, xml_path='supply_item_info.xml'):
        self.xml_path = xml_path
        self.dict_supply_items = {}
        self.ignore_ids = {}
        self.get_all_supply_items()

    def get_all_supply_items(self):
        DOMTree_supply_item = xml.dom.minidom.parse(self.xml_path)
        collection_supply_item = DOMTree_supply_item.documentElement
        for item_ in collection_supply_item.getElementsByTagName("supply_item"):
            supply_item = SupplyItem()
            supply_item.update(item_)
            self.dict_supply_items[supply_item.id] = supply_item

    def __len__(self):
        return len(self.dict_supply_items)

    def __getitem__(self, item):
        return self.dict_supply_items[item]

    def keys(self):
        return self.dict_supply_items.keys()

    def values(self):
        return self.dict_supply_items.values()

    def items(self):
        return self.dict_supply_items.items()


all_supply_items = AllSupplyItems(xml_path=os.path.join(os.path.dirname(__file__), 'supply_item_info.xml'))

if __name__ == '__main__':
    print(len(all_supply_items))
    print(all_supply_items.values())
