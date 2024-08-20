import os
import xml
from xml.dom.minidom import parse


def check_int(value, revalue=0, debug=False):
    if value is None or value == "":
        if debug:
            print(f'value is {value},i.e. None')
        return revalue
    return int(value)


class Character:
    item_arr = ["index","id", "name", "skill1", "skill2", "skill3"]

    def __init__(self):
        for item in self.item_arr:
            self.__dict__[item] = None

    def update(self, item):
        self.index = check_int(item.getAttribute("index"))
        self.id = check_int(item.getAttribute("id"))
        self.name = item.getAttribute("name")
        self.skill1 = check_int(item.getAttribute("skill1"))
        self.skill2 = check_int(item.getAttribute("skill2"))
        self.skill3 = check_int(item.getAttribute("skill3"))
        self.skills = [self.skill1,self.skill2, self.skill3]

    def __repr__(self):
        return f"id:{self.id}, name:{self.name},"


class AllCharacters:
    def __init__(self, xml_path='character_info.xml'):
        self.xml_path = xml_path
        self.dict_characters = {}
        self.ignore_ids = {}
        self.get_all_characters()

    def get_all_characters(self):
        DOMTree_character = xml.dom.minidom.parse(self.xml_path)
        collection_character = DOMTree_character.documentElement
        for item_ in collection_character.getElementsByTagName("character"):
            character = Character()
            character.update(item_)
            self.dict_characters[character.id] = character

    def __len__(self):
        return len(self.dict_characters)

    def __getitem__(self, item):
        return self.dict_characters[item]

    def keys(self):
        return self.dict_characters.keys()

    def values(self):
        return self.dict_characters.values()

    def items(self):
        return self.dict_characters.items()


all_characters = AllCharacters(xml_path=os.path.join(os.path.dirname(__file__), 'character_info.xml'))

if __name__ == '__main__':
    print(len(all_characters))
    print(all_characters.values())
