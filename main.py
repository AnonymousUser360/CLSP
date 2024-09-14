import copy
import json
import os
import gc
import random
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from transformers import DistilBertTokenizer
from transformers import AutoModel
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as CFG
from dataset import CLIPDataset, get_transforms
from CLIP import CLIPModel
from utils_clip import AvgMeter, get_lr,extract_Segments_KeyPoints
import gc


inspect_dict = {
        'My states':['health_point','position','speed','alive_state', 'view_direction','position_type','use_which_weapon'],
        'Nearest enemy states':['health_point','position','relative_position_to_me','distance_to_me','enemy_can_see_me','i_can_see_enemy','which_direction_enemy_is_located_to_me'],
        'Nearest teammate states':['health_point','position','relative_position_to_me','distance_to_me','which_direction_teammate_is_located_to_me'],
    }

def load_json(file_path):
    with open(file_path, 'r') as file:
       json_data = json.load(file)
    return json_data

def load_jsons(data_names):
    data_list = []
    for data in data_names:
        json_data=load_json(os.path.join(CFG.captions_path,f"4P_Align_Part{data}.json"))
        data_list.extend(json_data)
    return data_list

 

def make_train_valid_dfs(): 
    dataframe = load_jsons(CFG.all_data_parts)
    max_id = len(dataframe) 
    state_ids = np.arange(0, max_id) 
    np.random.seed(5)
    valid_ids = torch.load(CFG.valid_idx)
 
    train_dataframe,valid_dataframe= [],[]
    for id in state_ids:
        if id in valid_ids:
            valid_dataframe.append(dataframe[id])
        else:
            train_dataframe.append(dataframe[id])
 
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode,weights=None):
 
    transforms = None
    dataset = CLIPDataset(
        dataframe,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    if weights is None:
        if mode == 'train':
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=CFG.batch_size,
                num_workers=CFG.num_workers,
                shuffle=True  ,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=CFG.batch_size_eval,
                num_workers=CFG.num_workers_eval,
                shuffle=False,
            )
    else:
        dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        sampler=weights
    )
    return dataloader

def put_on_device(  data):
    if isinstance(data,dict):
        for k,v in data.items():
            data[k] =  put_on_device(v)
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(CFG.device)

def train_epoch(model, tokenizer,train_loader, optimizer, lr_scheduler, step,epoch,valid_df,valid_loader,result_my_state,result_enemy_state,result_teammate_state):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for idx,batch in enumerate(tqdm_object): 
        new_batch = {}
        for k, v in batch.items(): 
            if k in ["input_ids", 'attention_mask',  ]:
                new_batch[k] = v.to(CFG.device)
            elif k=='state':            
                new_batch[k] = put_on_device(v) 
        loss = model(new_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch['input_ids'].shape[0]
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        loss_str = f"Train Epoch:{epoch}, step_batch:{idx}, loss:{loss.item()}"
        if idx%CFG.retrieval_frequency==1:
            valid_state_embeddings  = get_state_embeddings(model,valid_loader)
            result_my_state,result_enemy_state,result_teammate_state=valid_retrieval_recall_loss(model,tokenizer,valid_state_embeddings,valid_df,epoch,idx, result_my_state,result_enemy_state,result_teammate_state,filename=f"{CFG.log_path}/{CFG.exp_name}.txt")
            del valid_state_embeddings
            valid_state_embeddings  = None
        print(loss_str)
 

        if not os.path.exists(CFG.log_path):
            os.makedirs(CFG.log_path)
 

        filename = f"{CFG.log_path}/{CFG.exp_name}.txt"
 
        if not os.path.exists(filename): 
            with open(filename, 'w') as f:
                pass  

        with open(f"{CFG.log_path}/{CFG.exp_name}.txt", 'a+') as f:
            f.write(loss_str+'\n')
    return loss_meter,result_my_state,result_enemy_state,result_teammate_state


def valid_loss(model, valid_loader, epoch, step):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for idx,batch in enumerate(tqdm_object): 
        new_batch = {}
        for k, v in batch.items(): 
            if k in ["input_ids", 'attention_mask',  ]: 
                new_batch[k] = v.to(CFG.device)
            elif k=='state':              
                new_batch[k] = put_on_device(v)
 
        loss = model(new_batch)

 
        count = batch['input_ids'].shape[0]
 
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        loss_str = f"---Test Loss: train_epoch:{epoch}, train_step:{step} test_step:{idx}, loss:{loss.item()}"
        print(loss_str)
        with open(f"{CFG.log_path}/{CFG.exp_name}.txt", 'a+') as f:
            f.write(loss_str+'\n')     
        if idx>20:
            break   
    return loss_meter

def compute_position_difference(pos1,pos2):
    diff = np.linalg.norm(np.array(pos1)-np.array(pos2))
    return diff

def compute_distance_difference(dist1,dist2):
    diff = np.abs(int(dist1.split(' ')[0]) - int(dist2.split(' ')[0]))
    return diff
def compute_hp_difference(dist1,dist2):
    try:
        diff = np.abs(int(dist1.split('(')[0]) - int(dist2.split('(')[0]))
    except:
        return 'Bad'
    return diff

def compute_yaw_difference(dist1,dist2):
    try:
        diff = np.abs(int(dist1.split('yaw_')[1]) - int(dist2.split('yaw_')[1]))
    except:
        return 'Bad'   
    if diff>180:
        diff=360-diff
    return diff

def compute_bool_difference(dist1,dist2):
    try:
        diff = 0 if dist1==dist2 else 1
    except:
        return 'Bad'
    return diff
  

def compute_value_difference(v1, v2, k):
    if (k in v1) and (k in v2):
        if k in ['position','relative_position_to_me', 'speed']: 
            distance = compute_position_difference(v1[k],v2[k])
        elif k in ['distance_to_me']: 
            distance = compute_distance_difference(v1[k],v2[k])
        elif k in ['health_point']: 
            distance = compute_hp_difference(v1[k],v2[k])
        elif k in [ 'view_direction','which_direction_enemy_is_located_to_me','which_direction_teammate_is_located_to_me']: 
            distance = compute_yaw_difference(v1[k],v2[k])
        else:
            distance = compute_bool_difference(v1[k],v2[k]) 
    else:
        print(f"请增加对类型{k}的处理函数")
        return 'Bad'

    return distance


def compute_states_difference(state_query, state_retrieval):
  
    result_dict = defaultdict(list)
    query_dict = eval(state_query)
    retrieval_dict = eval(state_retrieval)
    for k,v  in inspect_dict.items():
        if k in query_dict.keys() and k in retrieval_dict.keys():
            for t in v:
                diff_val = compute_value_difference(query_dict[k], retrieval_dict[k], t)
                result_dict[k].append(diff_val)

    return result_dict 

def update_result(result,inspect_dict,result_my_state,result_enemy_state,result_teammate_state):
    for k,v in result.items():
 
        if k=='My states':
            for idx,t in enumerate(inspect_dict[k]):
                result_my_state[t].append(v[idx])
        elif k=='Nearest enemy states':
            for idx,t in enumerate(inspect_dict[k]):
                result_enemy_state[t].append(v[idx])
        elif k=="Nearest teammate states":
            for idx,t in enumerate(inspect_dict[k]):
                result_teammate_state[t].append(v[idx])

    return result_my_state,result_enemy_state,result_teammate_state

def find_match_states(model, state_embeddings, query,  tokenizer,data_idx, RFF_encoding=False ):
     
    if RFF_encoding:
        state_description_token = query
        seg_dict = extract_Segments_KeyPoints(state_description_token,tokenizer ) 
        batch,new_batch = {},{}
        batch['input_ids']=torch.tensor(seg_dict['input_ids'])
        batch['attention_mask'] = torch.tensor(seg_dict['attention_mask'])
        batch['segments_length_pad'] = torch.tensor(seg_dict['segments_length_pad'])
        batch['key_values_unroll_pad']=torch.tensor(seg_dict['key_values_unroll_pad'])
        batch['key_values_segement_lens_pad']=torch.tensor(seg_dict['key_values_segement_lens_pad'])
        batch['segment_types'] = torch.tensor(seg_dict['segment_types'])
        for k, v in batch.items(): 
            if k in ["input_ids", 'attention_mask',  ]: 
                new_batch[k] = v.to(CFG.device)
            elif k=='state':     
                new_batch[k] = put_on_device(v)
            elif k in ['segments_length_pad','key_values_unroll_pad','key_values_segement_lens_pad','segment_types']:
                new_batch[k] = v  
        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=new_batch["input_ids"].view(1,-1), attention_mask=new_batch["attention_mask"].view(1,-1),segments_length_pad=new_batch["segments_length_pad"].view(1,-1),key_values_unroll_pad=new_batch["key_values_unroll_pad"].view(1,-1),key_values_segement_lens_pad=new_batch["key_values_segement_lens_pad"].view(1,-1),segment_types=new_batch['segment_types'].view(1,-1)
            )
            text_embeddings = model.text_projection(text_features)
    else:        
        encoded_query = tokenizer([query], padding=True, truncation=True, max_length=CFG.max_tokenizer_length)
        batch = {
            key: torch.tensor(values).to(CFG.device)
            for key, values in encoded_query.items()
        }
        new_batch = {}
        for k, v in batch.items(): 
            if k in ["input_ids", 'attention_mask',  ]: 
                new_batch[k] = v.to(CFG.device)
            elif k=='state':     
                new_batch[k] = put_on_device(v)

        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=new_batch["input_ids"], attention_mask=new_batch["attention_mask"]
            )
            text_embeddings = model.text_projection(text_features)
        
        
    image_embeddings_n = F.normalize(state_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    similaritys, indices = torch.topk(dot_similarity.squeeze(0),  10)
    top1 = data_idx in indices.cpu().numpy()[:1]
    top5 = data_idx in indices.cpu().numpy()[:5]
    top10 = data_idx in indices.cpu().numpy() 
    
    return similaritys,indices,top1,top5,top10

def valid_retrieval_recall_loss(model,tokenizer, valid_state_embeddings, valid_df, epoch, step,result_my_state,result_enemy_state,result_teammate_state,filename):
   
    num_valid_samples = len(valid_df)
    query_idx_list = list(range(num_valid_samples))
    random.shuffle(query_idx_list)
    query_idx_list_selected = query_idx_list[::2]
    
    Query_Times,top1_num,top5_num,top10_num = 0,0,0,0
    for sub_idx,idx in enumerate(query_idx_list_selected): 
        Query_Times+=1
        jd = valid_df[idx]
        query = jd["state_description"][0]
        query_clone = copy.deepcopy(query)
        query = query.replace("'",'').replace("_",' ') 
        similaritys,indices,top1,top5,top10 = find_match_states(model, valid_state_embeddings, query  ,tokenizer, idx, RFF_encoding=False)        
        top1_num+=top1
        top5_num+=top5
        top10_num+=top10
        for sub_idx,i in enumerate(indices[:1]): 
            retrieval_state = valid_df[i.item()]['state_description'][0] 
            if sub_idx==0:
                result = compute_states_difference(query_clone,retrieval_state)
                result_my_state,result_enemy_state,result_teammate_state=update_result(result,inspect_dict,result_my_state,result_enemy_state,result_teammate_state)
    
    str_tops = f"Recall Epoch: {epoch}, step:{step}, top_1_5_10_ratio:{[top1_num/Query_Times,top5_num/Query_Times,top10_num/Query_Times]}"
    mean_list, median_list = [], []
    with open(filename, 'a+') as f:
        f.write(str_tops+'\n') 
        print(str_tops)
    for kk in inspect_dict['My states']: 
        mean_list.append(np.mean(result_my_state[kk]))
        median_list.append(np.median(result_my_state[kk])) 
    for kk in inspect_dict['Nearest enemy states']: 
        mean_list.append(np.mean(result_enemy_state[kk]))
        median_list.append(np.median(result_enemy_state[kk]))
        
    for kk in inspect_dict['Nearest teammate states']:        
        mean_list.append(np.mean(result_teammate_state[kk]))
        median_list.append(np.median(result_teammate_state[kk]))
    str_mean_loss = f"Retrieval mean Epoch: {epoch}, step:{step}, Myself_Enemy_Teammate: {mean_list}"
    str_median_loss = f"Retrieval median Epoch: {epoch}, step:{step}, Myself_Enemy_Teammate: {median_list}"
    print(str_mean_loss)
    with open(filename, 'a+') as f:
        f.write(str_mean_loss+'\n')             
        f.write(str_median_loss+'\n')       
    return result_my_state,result_enemy_state,result_teammate_state

def valid_epoch(model, valid_loader,epoch):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for idx,batch in enumerate(tqdm_object): 
        new_batch = {}
        for k, v in batch.items(): 
            if k in ["input_ids", 'attention_mask',  ]: 
                new_batch[k] = v.to(CFG.device)
            elif k=='state':      
                new_batch[k] = put_on_device(v) 
        loss = model(new_batch)
        count = batch['input_ids'].shape[0]
 
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        loss_str = f"---Test Epoch:{epoch}, step_batch:{idx}, loss:{loss.item()}"
        print(loss_str)
        with open(f"{CFG.log_path}/{CFG.exp_name}.txt", 'a+') as f:
            f.write(loss_str+'\n')        
    return loss_meter


def get_state_embeddings( model, valid_loader ):
 
    model.eval()
    
    valid_state_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            new_batch = {}
            for k, v in batch.items(): 
                if k in ["input_ids", 'attention_mask',  ]: 
                    new_batch[k] = v.to(CFG.device)
                elif k=='state':           
                    new_batch[k] = put_on_device(v)
 
            state_features = model.state_encoder(new_batch["state"]) 
            state_embeddings = model.state_projection(state_features)
            valid_state_embeddings.append(state_embeddings)
    model.train()
    return  torch.cat(valid_state_embeddings)

def main():    
    os.environ["CUDA_VISIBLE_DEVICES"]=CFG.gpu_id 
    train_df, valid_df = make_train_valid_dfs()
 
    weights = None
    print(f"Loading tokenizer ...")
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    print(f"Building train loader ...")
    train_loader = build_loaders(train_df, tokenizer, mode="train", weights=weights)
    print(f"Building test loader ...")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    

    result_my_state = {'position':[],'speed':[],'health_point':[],'alive_state':[],'view_direction':[],'position_type':[],'use_which_weapon':[]}
    result_enemy_state = {'position':[],'relative_position_to_me':[],'distance_to_me':[],'health_point':[],'enemy_can_see_me':[],'i_can_see_enemy':[],'which_direction_enemy_is_located_to_me':[]}
    result_teammate_state = {'position':[],'relative_position_to_me':[],'distance_to_me':[],'health_point':[],'which_direction_teammate_is_located_to_me':[]}

 
    if CFG.device.type=='cuda':        
        torch.backends.cudnn.enabled = False

    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf') 
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
 
        model.train()
        train_loss,result_my_state,result_enemy_state,result_teammate_state = train_epoch(model,tokenizer, train_loader, optimizer, lr_scheduler, step, epoch,valid_df, valid_loader,result_my_state,result_enemy_state,result_teammate_state)
        model.eval() 
        gc.collect()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, epoch)
        
        if train_loss.avg < best_loss:
            best_loss = train_loss.avg 
            torch.save(model.state_encoder.state_dict(), f"/aaa/code/OpenAI-CLIP-master/ckpts/{CFG.exp_name}-StateEncoer-epoch{epoch}.pt")
            torch.save(model.state_projection.state_dict(), f"/aaa/code/OpenAI-CLIP-master/ckpts/{CFG.exp_name}-StateProjector-epoch{epoch}.pt")
            torch.save(model.text_encoder.state_dict(), f"/aaa/code/OpenAI-CLIP-master/ckpts/{CFG.exp_name}-TextEncoer-epoch{epoch}.pt")
            torch.save(model.text_projection.state_dict(), f"/aaa/code/OpenAI-CLIP-master/ckpts/{CFG.exp_name}-TextProjector-epoch{epoch}.pt")    
            print("Saved Best Model!")


if __name__ == "__main__":
    main()
