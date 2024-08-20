import torch,os
gpu_id = '0'

exp_name = '20P_State_RFF_1x2MLP_T'
 
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
 
debug = True
data_path = "/mnt/nfs2/aaa/FPS/Data_Parts"

data_parts = list(range(1,21))
all_data_parts = list(range(1,21))
 
valid_idx = '/mnt/nfs2/aaa/code/stateclip/valid_ids_new.pt'
 
captions_path = "/mnt/nfs2/aaa/FPS/jsons_new"
label_path = "/mnt/nfs2/aaa/FPS/Label"
log_path = '/mnt/nfs2/aaa/code/stateclip/log'
retrieval_log_path = '/mnt/nfs2/aaa/code/stateclip/retrieval_log'
batch_size = 125
num_workers = 4
batch_size_eval = 64
num_workers_eval = 2
lr = 1e-4
weight_decay = 1e-4
patience = 2
factor = 0.5
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model_name = 'resnet50'
image_embedding = 2048
model_name = 'mlp'
state_embedding =  4096# 256#4096  RL
state_hidden_size = 256
text_encoder_model = "/mnt/nfs2/aaa/code/stateclip/distilbert-base-uncased"
text_expand_embedding = 4096# # 256#4096 RL
text_embedding =  768
text_hidden_size = 512
text_tokenizer = "/mnt/nfs2/aaa/code/stateclip/distilbert-base-uncased"
max_tokenizer_length = 512
 
retrieval_frequency = 2000
text_projector_name = ''
text_encoder_name = ''
state_projector_name = ''
state_encoder_name = ''
image_pretrained = False # for both image encoder and text encoder
state_pretrained = False # for both image encoder and text encoder
text_pretrained = True # for both image encoder and text encoder
image_trainable = False # for both image encoder and text encoder
state_trainable = False # for both image encoder and text encoder
text_trainable = False # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256
dropout = 0.1
tokenizer_stop_num = 102
RFF_dim  = 512
RFF_sigma  = 1
inference_exp_name = '20P_SEPretrain20P_Epoch3_NerfEncoding-epoch1'
inference_data_part = [21  ]#list(range(5,16))
inference_query_data_part = [21 ] #list(range(1,2))
