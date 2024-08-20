import os
import sys
import cv2
# from tomlkit import key_value
import torch
import albumentations as A
import clip
import config as CFG
from utils_clip import  extract_Segments_KeyPoints_simple


sys.path.insert(0, '/mnt/nfs2/aaa/Process_Replay/')

from load_trajectory_example import trajectory 

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        #"state","state_description"
        self.state_filenames = []
        self.state_description  = []
        self.tokenizer = tokenizer
 
        for data in dataframe:
            self.state_filenames.append(data["state"])        
            self.state_description.append(data["state_description"][0].replace("'",'').replace("_",' '))
 

        self.transforms = transforms
        self.trajectory_agent=trajectory(config_path='./utils/user_config_replay.yaml')
    def squeeze_wj(self, data):
        if isinstance(data,dict):
            for k,v in data.items():
                data[k] = self.squeeze_wj(v)
            return data
        elif isinstance(data, torch.Tensor):
            return data.squeeze(dim=0)
        
    def __getitem__(self, idx):
        item = {}
         
        pt_file =  f"{CFG.data_path}/{self.state_filenames[idx]}"
        state_file = pt_file.split(':')[0]
        player_id = pt_file.split(':')[1]
        pt_content = torch.load(state_file)
        self.trajectory_agent.features.id =  int(player_id)
        model_input = self.trajectory_agent.load_trajectory_onestep(load_pt=True,pt_content=pt_content)        
        model_input_squeeze = self.squeeze_wj(model_input)
 
        item['state'] =  model_input_squeeze 
        state_description  =  self.state_description[idx]
        seg_dict = extract_Segments_KeyPoints_simple(state_description ,self.tokenizer ) 
        item['input_ids']=torch.tensor(seg_dict['input_ids']).squeeze(dim=0)
        item['attention_mask']=torch.tensor(seg_dict['attention_mask']).squeeze(dim=0)
             
        return item


    def __len__(self):
        return len(self.state_description )


class CLIPDataset_image(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        #"state","state_description"
        self.image_filenames = []
        self.captions = []
        self.encoded_captions = []
        captions_list =[]
        for data in dataframe:
            self.image_filenames.append(data["image"]) 
            self.captions.append(data["state_description_notyaw_self"])
            captions_list.append(data["state_description_notyaw_self"][0])
        self.encoded_captions = tokenizer(captions_list, padding=True, truncation=True, max_length=CFG.max_length)
        self.transforms = transforms        

    def squeeze_wj(self, data):
        if isinstance(data,dict):
            for k,v in data.items():
                data[k] = self.squeeze_wj(v)
            return data
        elif isinstance(data, torch.Tensor):
            return data.squeeze(dim=0)
        
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx]).squeeze(dim=0)
            for key, values in self.encoded_captions.items()
        }
        image = cv2.imread(f"{CFG.data_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
     
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)
    

class CLIPDataset_image_cc3m(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        #"state","state_description"
        self.image_filenames = []
        self.captions = []
        self.encoded_captions = []
        captions_list =[]
        for data in dataframe:
            self.image_filenames.append(data["image"])
            self.captions.append(data["caption"])
            captions_list.append(data["caption"])
        self.encoded_captions = tokenizer(captions_list, padding=True, truncation=True, max_length=CFG.max_length)
        
    def squeeze_wj(self, data):
        if isinstance(data,dict):
            for k,v in data.items():
                data[k] = self.squeeze_wj(v)
            return data
        elif isinstance(data, torch.Tensor):
            return data.squeeze(dim=0)
        
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx]).squeeze(dim=0)
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = aspect_ratio_preserving_resize(image=image)
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
     
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)
    

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.CenterCrop(height=512, width=512, always_apply=True),
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.CenterCrop(height=512, width=512, always_apply=True),
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

def aspect_ratio_preserving_resize(image=None, target_size=(224, 224)):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if aspect_ratio < 1:  # 宽大于高
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # 高大于宽
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # 使用中心裁剪
    top = (new_height - target_size[1]) // 2
    bottom = new_height - top
    left = (new_width - target_size[0]) // 2
    right = new_width - left
    # cropped_image = resized_image[top:bottom, left:right]
    cropped_image = resized_image[top:top+target_size[0], left:left+target_size[1]]
    return cropped_image    
