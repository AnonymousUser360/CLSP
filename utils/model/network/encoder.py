import numpy as np
import torch
import torch.nn as nn


class OnehotEncoder(nn.Module):
    def __init__(self, num_embeddings: int):
        super(OnehotEncoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.main = nn.Embedding.from_pretrained(torch.eye(self.num_embeddings), freeze=True,
                                                 padding_idx=None)

    def forward(self, x: torch.Tensor):
        x = x.long().clamp_(max=self.num_embeddings - 1)
        return self.main(x)


class OnehotEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(OnehotEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.main = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

    def forward(self, x: torch.Tensor):
        x = x.long().clamp_(max=self.num_embeddings - 1)
        return self.main(x)


class BinaryEncoder(nn.Module):
    def __init__(self, num_embeddings: int):
        super(BinaryEncoder, self).__init__()
        self.bit_num = num_embeddings
        self.main = nn.Embedding.from_pretrained(self.get_binary_embed_matrix(self.bit_num), freeze=True,
                                                 padding_idx=None)

    @staticmethod
    def get_binary_embed_matrix(bit_num):
        embedding_matrix = []
        for n in range(2 ** bit_num):
            embedding = [n >> d & 1 for d in range(bit_num)][::-1]
            embedding_matrix.append(embedding)
        return torch.tensor(embedding_matrix, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        x = x.long().clamp_(max=2 ** self.bit_num - 1)
        return self.main(x)


class SignBinaryEncoder(nn.Module):
    def __init__(self, num_embeddings):
        super(SignBinaryEncoder, self).__init__()
        self.bit_num = num_embeddings
        self.main = nn.Embedding.from_pretrained(self.get_sign_binary_matrix(self.bit_num), freeze=True,
                                                 padding_idx=None)
        self.max_val = 2 ** (self.bit_num - 1) - 1

    @staticmethod
    def get_sign_binary_matrix(bit_num):
        neg_embedding_matrix = []
        pos_embedding_matrix = []
        for n in range(1, 2 ** (bit_num - 1)):
            embedding = [n >> d & 1 for d in range(bit_num - 1)][::-1]
            neg_embedding_matrix.append([1] + embedding)
            pos_embedding_matrix.append([0] + embedding)
        embedding_matrix = neg_embedding_matrix[::-1] + [[0 for _ in range(bit_num)]] + pos_embedding_matrix
        return torch.tensor(embedding_matrix, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        x = x.long().clamp_(max=self.max_val, min=- self.max_val)
        return self.main(x + self.max_val)


class PositionEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=None):
        super(PositionEncoder, self).__init__()
        self.n_position = num_embeddings
        self.embedding_dim = self.n_position if embedding_dim is None else embedding_dim
        self.position_enc = nn.Embedding.from_pretrained(
            self.position_encoding_init(self.n_position, self.embedding_dim),
            freeze=True, padding_idx=None)

    @staticmethod
    def position_encoding_init(n_position, embedding_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
            for pos in range(n_position)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # apply sin on 0th,2nd,4th...embedding_dim
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # apply cos on 1st,3rd,5th...embedding_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, x: torch.Tensor):
        return self.position_enc(x)


class TimeEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.position_array = torch.nn.Parameter(self.get_position_array(), requires_grad=False)

    def get_position_array(self):
        x = torch.arange(0, self.embedding_dim, dtype=torch.float)
        x = x // 2 * 2
        x = torch.div(x, self.embedding_dim)
        x = torch.pow(10000., x)
        x = torch.div(1., x)
        return x

    def forward(self, x: torch.Tensor):
        v = torch.zeros(size=(x.shape[0], self.embedding_dim), dtype=torch.float, device=x.device)
        assert len(x.shape) == 1
        x = x.unsqueeze(dim=1)
        v[:, 0::2] = torch.sin(x * self.position_array[0::2])  # even
        v[:, 1::2] = torch.cos(x * self.position_array[1::2])  # odd
        return v

class MultiScaleNormEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(MultiScaleNormEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.scale_array = torch.nn.Parameter(self.get_scale_array(), requires_grad=False)

    def get_scale_array(self):
        x = torch.arange(-self.embedding_dim,self.embedding_dim, dtype=torch.float)
        x = torch.pow(10., x) 
        return x

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 1:
            v = torch.zeros(size=(x.shape[0], 2*self.embedding_dim), dtype=torch.float, device=x.device)            
            x = x.unsqueeze(dim=1)
            v[:, :] = torch.tanh(x * self.scale_array)  
        elif len(x.shape) == 2:
            v = torch.zeros(size=(x.shape[0],x.shape[1],  2*self.embedding_dim ), dtype=torch.float, device=x.device)            
            x = x.unsqueeze(dim=-1)
            v[:,:, :] = torch.sin(x * self.scale_array)
        return v


class FrequencyEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(FrequencyEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.position_array = torch.nn.Parameter(self.get_position_array(), requires_grad=False)

    def get_position_array(self):
        x = torch.arange(0, self.embedding_dim*2, dtype=torch.float)
        x = x // 2  
        x = torch.pow(2., x) 
        return x

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 1:
            v = torch.zeros(size=(x.shape[0], self.embedding_dim*2), dtype=torch.float, device=x.device)            
            x = x.unsqueeze(dim=1)
            v[:, 0::2] = torch.sin(x * self.position_array[0::2])  # even
            v[:, 1::2] = torch.cos(x * self.position_array[1::2])  # odd
        elif len(x.shape) == 2:
            v = torch.zeros(size=(x.shape[0],x.shape[1],  self.embedding_dim*2), dtype=torch.float, device=x.device)            
            x = x.unsqueeze(dim=-1)
            v[:,:, 0::2] = torch.sin(x * self.position_array[0::2])  # even
            v[:,:, 1::2] = torch.cos(x * self.position_array[1::2])  # odd
        return v



class GaussianEncoder(nn.Module):
    def __init__(self, embedding_dim, sigma=1):
        super(GaussianEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.Hierachical = False
        self.CAPSULE = True
        self.gaussian_array = torch.nn.Parameter(self.get_gaussian_array(sigma=1), requires_grad=False)
        if self.Hierachical:
            self.gaussian_array_1 = torch.nn.Parameter(self.get_gaussian_array(sigma=0.1), requires_grad=False)
            self.gaussian_array_2 = torch.nn.Parameter(self.get_gaussian_array(sigma=0.5), requires_grad=False)
            # self.gaussian_array_3 = torch.nn.Parameter(self.get_gaussian_array(sigma=0.5), requires_grad=False)
            # self.gaussian_array_4 = torch.nn.Parameter(self.get_gaussian_array(sigma=0.75), requires_grad=False)
        
        if self.CAPSULE:
            self.capsule = nn.Sequential(nn.Linear(self.embedding_dim*2, 512),
                                        nn.ReLU(),
                                        #  nn.Linear(512, 512),
                                        #  nn.ReLU(),
                                        nn.Linear(512, self.embedding_dim*2),
                                        nn.ReLU())
        else:
            self.capsule = None    

    def get_gaussian_array(self,sigma=1):
        x = torch.randn((self.embedding_dim))*sigma
        return x

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 1:
            v = torch.zeros(size=(x.shape[0], self.embedding_dim*2), dtype=torch.float, device=x.device)                        
            x = x.unsqueeze(dim=1)
            vp = 2*np.pi*x*self.gaussian_array.T
            v[:, 0::2] = torch.sin(vp)  # even
            v[:, 1::2] = torch.cos(vp)  # odd
            if self.Hierachical:
                v1 = torch.zeros(size=(x.shape[0], self.embedding_dim*2), dtype=torch.float, device=x.device)                        
                vp_1 = 2*np.pi*x*self.gaussian_array_1.T
                v1[:, 0::2] = torch.sin(vp_1)  # even
                v1[:, 1::2] = torch.cos(vp_1)  # odd
                v2 = torch.zeros(size=(x.shape[0], self.embedding_dim*2), dtype=torch.float, device=x.device)                        
                vp_2 = 2*np.pi*x*self.gaussian_array_2.T
                v2[:, 0::2] = torch.sin(vp_2)  # even
                v2[:, 1::2] = torch.cos(vp_2)  # odd   
    
        elif len(x.shape) == 2:
            v = torch.zeros(size=(x.shape[0],x.shape[1],  self.embedding_dim*2), dtype=torch.float, device=x.device)            
            x = x.unsqueeze(dim=-1)
            vp = 2*np.pi*x*self.gaussian_array 
            v[:,:, 0::2] = torch.sin(vp)  # even
            v[:,:, 1::2] = torch.cos(vp)  # odd
            if self.Hierachical:
                v1 =torch.zeros(size=(x.shape[0],x.shape[1],  self.embedding_dim*2), dtype=torch.float, device=x.device)                                  
                # print(f"x shape:{x.shape}, self.gaussian_array:{self.gaussian_array.shape}, self.gaussian_array_1:{self.gaussian_array_1.shape},v:{v.shape},v1:{v1.shape}")
                vp_1 = 2*np.pi*x*self.gaussian_array_1 
                v1[:, :,0::2] = torch.sin(vp_1)  # even
                v1[:, :,1::2] = torch.cos(vp_1)  # odd
                v2 =torch.zeros(size=(x.shape[0],x.shape[1],  self.embedding_dim*2), dtype=torch.float, device=x.device)                                                     
                vp_2 = 2*np.pi*x*self.gaussian_array_2 
                v2[:, :,0::2] = torch.sin(vp_2)  # even
                v2[:, :,1::2] = torch.cos(vp_2)  # odd 
        if self.CAPSULE:
            if self.Hierachical:
                v = self.capsule(v)
                v1 = self.capsule(v1)
                v2 = self.capsule(v2)
                v_ret = (v+v1+v2 )/3.
            else:
                v_ret = self.capsule(v)
        else:
            v_ret = v
    
        return v_ret

 
    
class UnsqueezeEncoder(nn.Module):
    def __init__(self, unsqueeze_dim: int = -1, norm_value: float = 1):
        super(UnsqueezeEncoder, self).__init__()
        self.unsqueeze_dim = unsqueeze_dim
        self.norm_value = norm_value

    def forward(self, x: torch.Tensor):
        x = x.float().unsqueeze(dim=self.unsqueeze_dim)
        if self.norm_value != 1:
            x = x / self.norm_value
        return x


if __name__ == '__main__':
    pass
