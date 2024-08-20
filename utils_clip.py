from collections import defaultdict
import copy
import json
import os
import re
from typing import NoReturn, Optional, List
import config as CFG
import yaml
from easydict import EasyDict


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def read_config(path: str) -> EasyDict:
    """
    Overview:
        read configuration from path
    Arguments:
        - path (:obj:`str`): Path of source yaml
    Returns:
        - (:obj:`EasyDict`): Config data from this file with dict type
    """
    if path:
        assert os.path.exists(path), path
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return EasyDict(config)


def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Overview:
        merge two dict using deep_update
    Arguments:
        - original (:obj:`dict`): Dict 1.
        - new_dict (:obj:`dict`): Dict 2.
    Returns:
        - (:obj:`dict`): A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:  # if new_dict is neither empty dict nor None
        deep_update(merged, new_dict, True, [])

    return merged


def deep_update(
    original: dict,
    new_dict: dict,
    new_keys_allowed: bool = False,
    whitelist: Optional[List[str]] = None,
    override_all_if_type_changes: Optional[List[str]] = None
):
    """
    Overview:
        Updates original dict with values from new_dict recursively.

    .. note::

        If new key is introduced in new_dict, then if new_keys_allowed is not
        True, an error will be thrown. Further, for sub-dicts, if the key is
        in the whitelist, then new subkeys can be introduced.

    Arguments:
        - original (:obj:`dict`): Dictionary with default values.
        - new_dict (:obj:`dict`): Dictionary with values to be updated
        - new_keys_allowed (:obj:`bool`): Whether new keys are allowed.
        - whitelist (Optional[List[str]]): List of keys that correspond to dict
            values where new subkeys can be introduced. This is only at the top
            level.
        - override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (:obj:`dict`), if the "type" key in that value dict changes.
    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise RuntimeError("Unknown config parameter `{}`. Base config have: {}.".format(k, original.keys()))

        # Both original value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def padding_list(unpad_list, total_len = 50, padding_value=0):
    if len(unpad_list)>50:
        print(f"增大padding长度！！！")
        raise
    if padding_value=='':
        pad_list = unpad_list + (total_len-len(unpad_list))*['']
    else:
        pad_list = unpad_list + (total_len-len(unpad_list))*[0]

    return pad_list

def extract_Segments_KeyPoints(state_description, tokenizer, pattern=None):
    if pattern is None:
        pattern  = re.compile(r'<(.*?)>(.*?)</\1>')

    key_positions =[]
    key_values =[]
    key_values_unroll =[] 
    key_values_segement_lens =[] 
    key_names = []
    segments = []
    segment_types = []
    segments_length = []
    last_match_pos = 0
    # segment_type_dict = defaultdict(int)
    segment_type_dict = {}
    segment_type_dict['MY HP']=5
    segment_type_dict['MY ALIVE STATE']=0
    segment_type_dict['MY POS']=1
    segment_type_dict['MY SPEED']=2
    segment_type_dict['MY VIEW']=5
    segment_type_dict['MY POS TYPE']=0
    segment_type_dict['OBSTACLE']=4
    segment_type_dict['MY WEAPON']=0
    segment_type_dict['ENEMY HP']=5
    segment_type_dict['ENEMY SEE ME']=0
    segment_type_dict['ENEMY BE SEEN']=0
    segment_type_dict['ENEMY DISTANCE']=6
    segment_type_dict['ENEMY POS']=1
    segment_type_dict['ENEMY RELATIVE']=3
    segment_type_dict['ENEMY DIRECTION']=5
    segment_type_dict['TEAMMATE HP']=5
    segment_type_dict['TEAMMATE DISTANCE']=6
    segment_type_dict['TEAMMATE POS']=1
    segment_type_dict['TEAMMATE RELATIVE']=3
    segment_type_dict['TEAMMATE DIRECTION']=5
    segment_type_dict['DOOR POS']=1
    segment_type_dict['DOOR RELATIVE']=3
    segment_type_dict['DOOR DISTANCE']=6
    segment_type_dict['DAMAGE POS']=1
    segment_type_dict['DAMAGE RELATIVE']=3
    segment_type_dict['DAMAGE DISTANCE']=6    
 
    # 使用正则表达式查找所有匹配项
    matches = pattern.findall(state_description)
    for i, (var_name, var_value) in enumerate(matches):
        if i==0:
            start_pos = 0
        key_name = f"<{var_name}>{var_value}</{var_name}>"
        key_names.append(key_name)
        segment_types.append(segment_type_dict[var_name])
        length = len(key_name)
        pos = state_description.find(key_name)
        if '(' in var_value or ')' in var_value:
            var_value_trans = list(eval(var_value))
        else:
            var_value_trans = [round(float(var_value))]
        key_values.append(var_value_trans)
        key_values_unroll.extend(var_value_trans)
        key_values_segement_lens.append(len(var_value_trans))
        # key_values.append( [eval(var_value)] if type(eval(var_value))==int else list(eval(var_value)) )
        key_positions.append(i)   
        segments.append( state_description[start_pos:pos] ) 
        
        # print(f"segment {i+1}:{state_description[start_pos:pos]}, key value:{var_value}")
        start_pos = pos+length
        last_match_pos = start_pos

    segments.append( state_description[last_match_pos:] )   
    segment_num = len(segments)
    raw_token = tokenizer(segments, padding=True, truncation=True, max_length=CFG.max_tokenizer_length)
    input_ids, attention_mask = [], []
    for idx,sub_input_ids in enumerate(raw_token['input_ids']):
        if idx==0:
            start_idx = 0
            end_idx = sub_input_ids.index(CFG.tokenizer_stop_num)
            input_ids.extend(sub_input_ids[start_idx:end_idx])
            segments_length.append(end_idx-start_idx)
        elif idx==segment_num-1:
            start_idx = 1
            end_idx = sub_input_ids.index(CFG.tokenizer_stop_num)+1
            input_ids.extend(sub_input_ids[start_idx:end_idx])
            segments_length.append(end_idx-start_idx)
        else:
            start_idx = 1
            end_idx = sub_input_ids.index(CFG.tokenizer_stop_num) 
            input_ids.extend(sub_input_ids[start_idx:end_idx])
            segments_length.append(end_idx-start_idx)
    input_ids_length = len(input_ids)
    if input_ids_length<=CFG.max_tokenizer_length:
        input_ids = input_ids + (CFG.max_tokenizer_length - input_ids_length)*[0]
        attention_mask = [1]*input_ids_length + [0]*(CFG.max_tokenizer_length - input_ids_length)
    else:
        print(f"请增大CFG.max_tokenizer_length参数，满足不遗漏state描述文本编码内容")
        raise
    
    # segments_pad = padding_list(segments,total_len=50)
    segments_length_pad = padding_list(segments_length,total_len=50)
    key_values_unroll_pad = padding_list(key_values_unroll,total_len=50)
    key_values_segement_lens_pad = padding_list(key_values_segement_lens,total_len=50)
    segment_types_pad = padding_list(segment_types,total_len=50,padding_value=0)
    return {'input_ids':input_ids,'attention_mask':attention_mask,'segments_length_pad':segments_length_pad,
            'key_values_unroll_pad':key_values_unroll_pad,'key_values_segement_lens_pad':key_values_segement_lens_pad,
            'segments':segments,'segment_types':segment_types_pad}
    #, 'key_values':key_values, 'key_names':key_names


def extract_Segments_KeyPoints_simple(state_description, tokenizer, pattern=None):

    raw_token = tokenizer([state_description], padding=True, truncation=True, max_length=CFG.max_tokenizer_length)
    input_ids = raw_token['input_ids']
    attention_mask = raw_token['attention_mask']
 
    input_ids_length = len(input_ids[0])
    if input_ids_length<=CFG.max_tokenizer_length:
        input_ids = [input_ids[0] + (CFG.max_tokenizer_length - input_ids_length)*[0]]
        attention_mask = [[1]*input_ids_length + [0]*(CFG.max_tokenizer_length - input_ids_length)]
    else:
        print(f"请增大CFG.max_tokenizer_length参数，满足不遗漏state描述文本编码内容")
        # raise
    
 
    return {'input_ids':input_ids,'attention_mask':attention_mask }
 