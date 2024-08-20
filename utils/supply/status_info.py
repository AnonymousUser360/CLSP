# -*- coding:utf-8 -*
from enum import Enum, unique


@unique
class StatusType(Enum):
    ERROR = 0
    RESET = 1
    MOVE = 2
    ARRIVE = 3
    BEFORE_PICK = 4
    PICK = 5
    AFTER_PICK = 6
    STUCK = 7   # not enter supply module once
    DONE = 8    # not enter supply module any more


if __name__ == '__main__':
    print(StatusType(0).name, StatusType(0).value)
    print(StatusType['RESET'].name, StatusType['RESET'].value)
