import copy
import os
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import torch
from transformers import DistilBertTokenizer
import config as CFG 
from CLIP import CLIPModel 
 
 
inspect_dict = {
        'My states':['health_point','position','speed','alive_state', 'view_direction','position_type','use_which_weapon'],
        'Nearest enemy states':['health_point','position','relative_position_to_me','distance_to_me','enemy_can_see_me','i_can_see_enemy','which_direction_enemy_is_located_to_me'],
        'Nearest teammate states':['health_point','position','relative_position_to_me','distance_to_me','which_direction_teammate_is_located_to_me'],
    }

 
 

def put_on_device(  data):
    if isinstance(data,dict):
        for k,v in data.items():
            data[k] =  put_on_device(v)
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(CFG.device)

def train_epoch(model, tokenizer,result_my_state,result_enemy_state,result_teammate_state, state_file, state_description_file):

    valid_state_embeddings  = get_state_embeddings(model,state_file )
    result_my_state,result_enemy_state,result_teammate_state=valid_retrieval_recall_loss(model,tokenizer,valid_state_embeddings,state_description_file, result_my_state,result_enemy_state,result_teammate_state,filename=f"{CFG.log_path}/{CFG.exp_name}.txt")
    del valid_state_embeddings
    valid_state_embeddings  = None



    if not os.path.exists(CFG.log_path):
        os.makedirs(CFG.log_path)


    filename = f"{CFG.log_path}/{CFG.exp_name}.txt"
 
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            pass   
 
    return  result_my_state,result_enemy_state,result_teammate_state

 
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

def find_match_states(model, state_embeddings, query,  tokenizer,data_idx,   ):
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

def valid_retrieval_recall_loss(model,tokenizer, valid_state_embeddings, state_description_file, result_my_state,result_enemy_state,result_teammate_state,filename):
   
    num_valid_samples = len(state_description_file)
    query_idx_list = list(range(num_valid_samples))

    Query_Times,top1_num,top5_num,top10_num = 0,0,0,0
    for sub_idx,idx in enumerate(query_idx_list):
 
        Query_Times+=1
        query = state_description_file[idx][0]
 
        query_clone = copy.deepcopy(query)
        query = query.replace("'",'').replace("_",' ')

        similaritys,indices,top1,top5,top10 = find_match_states(model, valid_state_embeddings, query  ,tokenizer, idx)        
        top1_num+=top1
        top5_num+=top5
        top10_num+=top10
        for sub_idx,i in enumerate(indices[:1]):
   
            retrieval_state = state_description_file[i.item()][0]
 
            if sub_idx==0:
                result = compute_states_difference(query_clone,retrieval_state)
                result_my_state,result_enemy_state,result_teammate_state=update_result(result,inspect_dict,result_my_state,result_enemy_state,result_teammate_state)
    
    str_tops = f"Top_1_5_10_ratio:{[top1_num/Query_Times,top5_num/Query_Times,top10_num/Query_Times]}"
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
    str_mean_loss = f"Retrieval mean  Myself_Enemy_Teammate: {mean_list}"
    # str_median_loss = f"Retrieval median Epoch: {epoch}, step:{step}, Myself_Enemy_Teammate: {median_list}"
    print(str_mean_loss)
    with open(filename, 'a+') as f:
        f.write(str_mean_loss+'\n')             
        # f.write(str_median_loss+'\n')       
    return result_my_state,result_enemy_state,result_teammate_state
  
def get_state_embeddings( model,state_file  ):
 
    model.eval()
    valid_state_embeddings = []
    for state in state_file:         
        new_state = put_on_device(state)
        state_features = model.state_encoder(new_state) 
        state_embeddings = model.state_projection(state_features)
        valid_state_embeddings.append(state_embeddings)

    return  torch.cat(valid_state_embeddings)

def main():    
    os.environ["CUDA_VISIBLE_DEVICES"]=CFG.gpu_id
  
    print(f"Loading tokenizer ...")
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

    print(f"Loading state ...")
    state_file = torch.load('./data/modelinput_list.pt')
    state_description_file = torch.load('data/state_description_list.pt')

    result_my_state = {'position':[],'speed':[],'health_point':[],'alive_state':[],'view_direction':[],'position_type':[],'use_which_weapon':[]}
    result_enemy_state = {'position':[],'relative_position_to_me':[],'distance_to_me':[],'health_point':[],'enemy_can_see_me':[],'i_can_see_enemy':[],'which_direction_enemy_is_located_to_me':[]}
    result_teammate_state = {'position':[],'relative_position_to_me':[],'distance_to_me':[],'health_point':[],'which_direction_teammate_is_located_to_me':[]}

 

    if CFG.device.type=='cuda':        
        torch.backends.cudnn.enabled = False

    model = CLIPModel().to(CFG.device)
    model_base_path = './models/'
    state_encoder_path  = os.path.join(model_base_path,CFG.state_encoder_name)
    state_encoder = torch.load(state_encoder_path)
    state_projector_path  = os.path.join(model_base_path,CFG.state_projector_name)
    state_projector = torch.load(state_projector_path)
    text_encoder_path  = os.path.join(model_base_path,CFG.text_encoder_name)
    text_encoder = torch.load(text_encoder_path)
    text_projector_path  = os.path.join(model_base_path,CFG.text_projector_name)
    text_projector = torch.load(text_projector_path)    
    model.state_encoder.load_state_dict(state_dict=state_encoder)
    model.state_projection.load_state_dict(state_dict=state_projector)
    model.text_encoder.load_state_dict(state_dict=text_encoder)
    model.text_projection.load_state_dict(state_dict=text_projector)
    result_my_state,result_enemy_state,result_teammate_state = train_epoch(model,tokenizer,   result_my_state,result_enemy_state,result_teammate_state, state_file, state_description_file)
 

if __name__ == "__main__":
    main()
