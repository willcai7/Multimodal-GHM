import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import torch.nn.functional as F
import random 
import os 


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def generate_mask(n_token, n_i_token, batch_size, device='cpu'):
    # generate mask for auto-regressive model
    n_t_token = n_token - n_i_token
    mask = torch.zeros(n_token, n_token).to(device)
    mask[:n_i_token, n_i_token:] = float('-inf')
    mask[n_i_token:, n_i_token:] = torch.triu(torch.ones(n_t_token, n_t_token)*float('-inf'),diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, n_token, n_token)
    return mask

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, epsilon: float=1e-5):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = epsilon
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        rms_x = torch.sqrt(torch.square(x).mean(dim = -1, keepdim = True) + self.eps)
        x = x / rms_x * self.weight
        return x

class GELU(nn.Module): 
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return 0.5 * x * (1 + torch.erf(x / np.sqrt(2)))
    
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = GELU()
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, residual_pdrop: float | None = None):
        super(ResNetBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.residual_pdrop = residual_pdrop
        
        self.ln = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.drop = nn.Dropout(residual_pdrop)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x + self.drop(self.ffn(self.ln(x)))


class ResNet(nn.Module): 
    def __init__(self, d_model: int, d_ff: int, num_layers: int, num_classes:int, residual_pdrop: float | None):
        super(ResNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.residual_pdrop = residual_pdrop
        self.num_layers = num_layers
        
        # self.token_embeddings = nn.Embedding(2, d_model)
        self.layers = nn.ModuleList([ResNetBlock(d_model, d_ff, residual_pdrop) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        if num_classes ==2:
            self.rn_head = nn.Linear(d_model, 1, bias=False)
        else:
            self.rn_head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, x) -> torch.FloatTensor:
        x = x.to(torch.float)        # x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.rn_head(x)
        return x
    



def get_activation(activation="softmax"): # get activation function
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    elif activation == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    else:
        raise NotImplementedError

class AutoRegressiveTransformer(nn.Module):
    def __init__(self, n_token=9,n_i_token=4, 
                num_class=10, 
                n_embd=128, 
                n_layer=12, 
                n_guided_layers=[3,3], 
                n_head=4, 
                n_mlp_hidden=512,
                activation="softmax",  
                mlp=True, 
                normalize_attn=True,
                auto_regressive=False,
                sequential=False, 
                layernorm=True, 
                guide=False):
        super().__init__()
        self.name = f"EncoderTF_embd={n_embd}_layer={n_layer}_head={n_head}"

        # Network structure configs
        # self.n_dims = n_dims # input dimension
        self.vocab_size = num_class # vocabulary size
        self.context_length = n_token # context length
        self.n_token = n_token # number of tokens
        self.n_i_token = n_i_token # number of t tokens
        self.n_embd = n_embd # embedding dimension
        self.n_head = n_head # number of heads
        self.n_layer = n_layer # number of layers
        self.n_mlp_hidden = n_mlp_hidden # number of hidden units in MLP
        self.sequential = sequential # whether to use sequential training
        self.activation = get_activation(activation) # activation function
        self.mlp = mlp # whether to use MLP

        # Normalization config
        self.normalize_attn = normalize_attn # whether to normalize attention
        self.layernorm = layernorm # whether to use layer normalization
        self.auto_regressive = auto_regressive # whether to use auto-regressive model

        # Guided training config
        self.guide = guide # whether to use guided layer
        self.n_t_guided_layer =  n_guided_layers[0] # number of tmodel guided layers
        self.n_i_guided_layer =  n_guided_layers[1]
        self.guided_layer_gap = n_layer // (n_guided_layers[0]*2 +1) # guided layer gap 


        # layers
        self.position_embeddings = nn.Embedding(self.context_length, self.n_embd) # position embeddings

        self._queries = nn.ModuleList() # query layers
        self._keys = nn.ModuleList() # key layers
        self._values = nn.ModuleList() # value layers
        self._mlps = nn.ModuleList() # MLP layers
        self._lns_1 = nn.ModuleList() # layer norm layers
        self._lns_2 = nn.ModuleList() # layer norm layers
        self.t_guided_layer_flag = [False] * n_layer # guided layer flag
        self.i_guided_layer_flag = [False] * n_layer # guided layer flag
        self.t_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.i_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        guided_layer_counter = 0 # guided layer counter
        for i in range(n_layer):
            self._queries.append(nn.Linear(n_embd, n_embd, bias=False))
            self._keys.append(nn.Linear(n_embd, n_embd, bias=False))
            self._values.append(nn.Linear(n_embd, n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd])) 
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(n_embd, n_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(n_mlp_hidden, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))

            if guide and guided_layer_counter<(self.n_t_guided_layer*2+1) and (i+1) % self.guided_layer_gap == 0: # guided layer flag initialization
                self.t_guided_layer_flag[i] = True
                if guided_layer_counter < self.n_i_guided_layer:
                    self.i_guided_layer_flag[i] = True
                if guided_layer_counter == self.n_t_guided_layer-1 and self.n_i_guided_layer < self.n_t_guided_layer:
                    self.i_guided_layer_flag[i] = True
                guided_layer_counter += 1


        self._read_out = nn.Linear(n_embd, self.vocab_size) # read out layer
        self._out = nn.Linear(n_token, 1) #output layer 

    def token_embeddings(self, xt, zi):

        embeddings = torch.zeros(zi.size(0), zi.size(1) + xt.size(1), self.n_embd).to(zi.device)
        if self.sequential:
            num_zero_columns = self.n_embd - self.vocab_size 
            zeros_mat = torch.zeros(zi.size(0),zi.size(1), num_zero_columns).to(zi.device)
            x2 = torch.cat([zi, zeros_mat], dim=2)
            # print(x2.shape)
            embeddings[:, 0,:] = x2[:,0,:]
        else: 
            x2 = self.i_embedding(zi)
            embeddings[:, :self.n_i_token,:] = x2
        
        embeddings[:, self.n_i_token:,:] = self.t_embedding(xt)
        return embeddings

    def forward(self, xt, zi):
        contract_dim = 1 # contract dimension
        B,T1 = xt.size() # batch size and position size
        T2 = zi.size(1) # batch size and position size
        if self.auto_regressive:
            self.mask = generate_mask(self.n_token, self.n_i_token, B, device=xt.device)
        T = T1 + T2
        positions = torch.arange(T, device=xt.device).expand(B, T) # position embeddings
        # print(self.token_embeddings(xt, zi).shape)
        # print(self.position_embeddings(positions).shape)
        H = self.token_embeddings(xt, zi) + self.position_embeddings(positions) # token embeddings + position embeddings 
        guided_counter = 0 # guided layer counter
        index_q = 0 # index of hd 
        index_h = (self.n_t_guided_layer+1)*self.vocab_size # index of qd
        index_u = (2*self.n_t_guided_layer+1)*self.vocab_size # index of ud
        index_i = 0 # index of id
        t_guided_layers = []
        i_guided_layers = []
        for (q, k, v, mlp, ln1, ln2, t_guided_layer_flag, i_guided_layer_flag) in zip(
            self._queries, self._keys, self._values, self._mlps, self._lns_1, self._lns_2, self.t_guided_layer_flag, self.i_guided_layer_flag
        ):

            if self.layernorm: # layer norm
                H1 = ln1(H)
                query = q(H1) # query layer
                key = k(H1) # key layer
                value = v(H1) # value layer
            else:
                query = q(H) # query layer
                key = k(H) # key layer
                value = v(H) # value layer

            attn_weights = torch.matmul(query, key.transpose(-2, -1)) 
            if self.auto_regressive:
                # print(self.mask.shape)
                attn_weights += self.mask
            if self.normalize_attn: # normalize attention weights
                attn_weights = attn_weights / np.sqrt(self.n_embd)
            attn_weights = self.activation(attn_weights)
            H = H + torch.einsum('bij,bjd->bid', attn_weights, value) # attention layer + residual connection 
            if self.normalize_attn: # normalize attention weights
                attn_weights = attn_weights / H.shape[2] 
            H = H + torch.einsum('bij,bjd->bid', attn_weights, value) # attention layer + residual connection 
            

            if self.layernorm:
                H2 = ln2(H)
                if self.mlp: # MLP layer 
                    H = H + mlp(H2) 

            else: 
                if self.mlp: # MLP layer
                    H = H + mlp(H)
            
            if self.guide and t_guided_layer_flag: # guided layer
                
                if guided_counter == 0: # leave nodes 
                    q_H = H[:,self.n_i_token:, index_q: index_q+self.vocab_size] # qd
                    index_q += self.vocab_size # update index_q
                    guide_output = q_H
                
                elif 0< guided_counter < self.n_t_guided_layer+1: # downward process 
                    h_H = H[:,self.n_i_token:, index_h: index_h+self.vocab_size] # hd 
                    q_H = H[:,self.n_i_token:, index_q: index_q+self.vocab_size] # qd 
                    index_h += self.vocab_size # update index_h
                    index_q += self.vocab_size # update index_q
                    guide_output = torch.concatenate([h_H, q_H], dim=2) # concatenate hd and qd
                else:
                    # print(guided_counter)
                    # print("Index_u", index_u)
                    u_H = H[:,self.n_i_token:, index_u: index_u+self.vocab_size] # ud
                    index_u += self.vocab_size # update index_u
                    guide_output = u_H 
                guided_counter += 1
                t_guided_layers.append(guide_output)
            
            if self.guide and i_guided_layer_flag: 
                i_H = H[:,:self.n_i_token, index_i: index_i+self.vocab_size] # id
                index_i += self.vocab_size
                i_guided_layers.append(i_H)
            
        prediction = self._read_out(H) # read out layer
        
        guided_layers = [t_guided_layers, i_guided_layers]
        return prediction[:,self.n_i_token:,:], guided_layers   

class ConditionalDenoiseEncoderTransformer(nn.Module):
    def __init__(self, n_token,n_i_token, num_class, n_embd=128, n_layer=12, n_guided_layers=[3,3], n_head=4, n_mlp_hidden=512,
                 activation="softmax",  mlp=True, normalize_attn=True,auto_regressive=False,
                sequential=False, layernorm=True, maxnorm=False, guide=False, sigma=1):
        super(ConditionalDenoiseEncoderTransformer, self).__init__()
        self.name = f"EncoderTF_embd={n_embd}_layer={n_layer}_head={n_head}"

        # Network structure configs
        # self.n_dims = n_dims # input dimension
        self.vocab_size = num_class # vocabulary size
        self.context_length = n_token # context length
        self.n_token = n_token # number of tokens
        self.n_i_token = n_i_token # number of t tokens
        self.n_embd = n_embd # embedding dimension
        self.n_head = n_head # number of heads
        self.n_layer = n_layer # number of layers
        self.n_mlp_hidden = n_mlp_hidden # number of hidden units in MLP
        self.sequential = sequential # whether to use sequential model
        self.activation = get_activation(activation) # activation function
        self.mlp = mlp # whether to use MLP

        # Normalization config
        self.normalize_attn = normalize_attn # whether to normalize attention
        self.layernorm = layernorm # whether to use layer normalization
        self.maxnorm = maxnorm # whether to use max normalization
        self.auto_regressive = auto_regressive # whether to use auto-regressive model

        # Guided training config
        self.guide = guide # whether to use guided layer
        self.n_t_guided_layer =  n_guided_layers[0] # number of tmodel guided layers
        self.n_i_guided_layer =  n_guided_layers[1]
        self.guided_layer_gap = n_layer // (n_guided_layers[1]*2 +1) # guided layer gap 


        # tree structure    
        # self.n_tree_layer = n_tree_layer # number of tree layers
        # self.n_tree_child = n_tree_child # number of tree children
        self.sigma = sigma # noise level

        # layers
        self.position_embeddings = nn.Embedding(self.context_length, self.n_embd) # position embeddings

        self._queries = nn.ModuleList() # query layers
        self._keys = nn.ModuleList() # key layers
        self._values = nn.ModuleList() # value layers
        self._mlps = nn.ModuleList() # MLP layers
        self._lns_1 = nn.ModuleList() # layer norm layers
        self._lns_2 = nn.ModuleList() # layer norm layers
        self.t_guided_layer_flag = [False] * n_layer # guided layer flag
        self.i_guided_layer_flag = [False] * n_layer # guided layer flag
        self.t_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        guided_layer_counter = 0 # guided layer counter
        for i in range(n_layer):
            self._queries.append(nn.Linear(n_embd, n_embd, bias=False))
            self._keys.append(nn.Linear(n_embd, n_embd, bias=False))
            self._values.append(nn.Linear(n_embd, n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd])) 
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(n_embd, n_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(n_mlp_hidden, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))

            if guide and guided_layer_counter<(self.n_i_guided_layer*2+1) and (i+1) % self.guided_layer_gap == 0: # guided layer flag initialization
                self.i_guided_layer_flag[i] = True
                if guided_layer_counter < self.n_t_guided_layer:
                    self.t_guided_layer_flag[i] = True
                if guided_layer_counter == self.n_i_guided_layer-1 and self.n_t_guided_layer < self.n_i_guided_layer:
                    self.t_guided_layer_flag[i] = True
                guided_layer_counter += 1


        self._read_out = nn.Linear(n_embd, 1) # read out layer
        self._out = nn.Linear(n_token, 1) #output layer 

    def token_embeddings(self, xt, zi):

        embeddings = torch.zeros(zi.size(0), zi.size(1) + xt.size(1), self.n_embd).to(zi.device)

        # print(x2.shape)
        leave_options = torch.arange(0, self.vocab_size).unsqueeze(0).unsqueeze(0).expand(zi.size(0), zi.size(1), self.vocab_size).to(zi.device)
        leave_options = -torch.pow(leave_options - zi.unsqueeze(-1),2)/2
        embeddings[:,:self.n_i_token,:self.vocab_size] = leave_options

        if self.sequential:
            num_zero_columns = self.n_embd - self.vocab_size 
            zeros_mat = torch.zeros(xt.size(0),xt.size(1), num_zero_columns).to(xt.device)
            x2 = torch.cat([xt, zeros_mat], dim=2)
            embeddings[:, self.n_i_token:,:] = x2
        else:
            x2 = self.t_embedding(xt)
            embeddings[:, self.n_i_token:,:] = x2
        return embeddings

    def forward(self, xt, zi):
        contract_dim = 1 # contract dimension
        T1 = xt.size(1) # batch size and position size
        B, T2 = zi.size() # batch size and position size
        if self.auto_regressive:
            self.mask = generate_mask(self.n_token, self.n_i_token, B, device=xt.device)
        T = T1 + T2
        positions = torch.arange(T, device=xt.device).expand(B, T) # position embeddings
        H = self.token_embeddings(xt, zi) + self.position_embeddings(positions) # token embeddings + position embeddings 
        guided_counter = 0 # guided layer counter
        index_h = 0 # index of hd 
        index_q = self.n_t_guided_layer*self.vocab_size # index of qd
        index_u = 2*self.n_t_guided_layer*self.vocab_size # index of ud
        index_i = 0 # index of id
        t_guided_layers = []
        i_guided_layers = []
        for (q, k, v, mlp, ln1, ln2, t_guided_layer_flag, i_guided_layer_flag) in zip(
            self._queries, self._keys, self._values, self._mlps, self._lns_1, self._lns_2, self.t_guided_layer_flag, self.i_guided_layer_flag
        ):

            if self.layernorm: # layer norm
                H1 = ln1(H)
                query = q(H1) # query layer
                key = k(H1) # key layer
                value = v(H1) # value layer
            else:
                query = q(H) # query layer
                key = k(H) # key layer
                value = v(H) # value layer

            attn_weights = torch.einsum('bid,bjd->bij', query, key) # attention weights
            if self.normalize_attn: # normalize attention weights
                attn_weights = attn_weights / np.sqrt(H.shape[2]) 
            if self.auto_regressive:
                attn_weights = attn_weights + self.mask
            attn_weights = self.activation(attn_weights)
            H = H + torch.einsum('bij,bjd->bid', attn_weights, value) # attention layer + residual connection 
            

            if self.layernorm:
                H2 = ln2(H)
                if self.maxnorm: 
                    H2 = H2 - torch.max(H2, dim=2)[0].unsqueeze(2)
                if self.mlp: # MLP layer 
                    H = H + mlp(H2) 

            else: 
                if self.maxnorm:
                    H = H - torch.max(H, dim=2)[0].unsqueeze(2)
                if self.mlp: # MLP layer
                    H = H + mlp(H)
            
            if self.guide and i_guided_layer_flag: # guided layer
                
                if guided_counter < self.n_i_guided_layer+1: # downward process 
                    h_H = H[:,:self.n_i_token, index_h: index_h+self.vocab_size] # hd 
                    q_H = H[:,:self.n_i_token, index_q: index_q+self.vocab_size] # qd 
                    index_h += self.vocab_size # update index_h
                    index_q += self.vocab_size # update index_q
                    guide_output = torch.concatenate([h_H, q_H], dim=2) # concatenate hd and qd
                else:
                    index_h -= self.vocab_size # update index_h
                    index_q -= self.vocab_size # update index_q
                    h_H = H[:,:self.n_i_token, index_h: index_h+self.vocab_size] # hd 
                    q_H = H[:,:self.n_i_token, index_q: index_q+self.vocab_size] # qd 
                    u_H = H[:,:self.n_i_token, index_u: index_u+self.vocab_size] # ud
                    index_u += self.vocab_size # update index_u
                    guide_output = torch.concatenate([h_H, q_H, u_H], dim=2) # concatenate hd, qd and ud
                guided_counter += 1
                i_guided_layers.append(guide_output)
            
            if self.guide and t_guided_layer_flag: 
                    i_H = H[:,self.n_i_token:, index_i: index_i+self.vocab_size] # id
                    index_i += self.vocab_size
                    t_guided_layers.append(i_H)
            
        prediction = self._read_out(H) # read out layer
        
        guided_layers = [t_guided_layers, i_guided_layers]
        return prediction[:,:self.n_i_token,0], guided_layers 

class DenoiseEncoderTransformer(nn.Module):
    def __init__(self, n_token, num_class, n_embd=128, n_layer=12, n_tree_layer=3, n_tree_child=3, n_guided_layer=3, n_head=4, n_mlp_hidden=512,
                 activation="softmax",  mlp=True, normalize_attn=True, layernorm=True, maxnorm=False, guide=False, sigma=1):
        super(DenoiseEncoderTransformer, self).__init__()
        self.name = f"EncoderTF_embd={n_embd}_layer={n_layer}_head={n_head}"

        # Network structure configs
        # self.n_dims = n_dims # input dimension
        self.vocab_size = num_class # vocabulary size
        self.context_length = n_token # context length
        self.n_token = n_token # number of tokens
        self.n_embd = n_embd # embedding dimension
        self.n_head = n_head # number of heads
        self.n_layer = n_layer # number of layers
        self.n_mlp_hidden = n_mlp_hidden # number of hidden units in MLP
        self.activation = get_activation(activation) # activation function
        self.mlp = mlp # whether to use MLP

        # Normalization config
        self.normalize_attn = normalize_attn # whether to normalize attention
        self.layernorm = layernorm # whether to use layer normalization
        self.maxnorm = maxnorm # whether to use max normalization

        # Guided training config
        self.guide = guide # whether to use guided layer
        self.n_guided_layer =  n_guided_layer # number of guided layers
        self.guided_layer_gap = n_layer // (n_guided_layer*2) # guided layer gap 


        # tree structure    
        self.n_tree_layer = n_tree_layer # number of tree layers
        self.n_tree_child = n_tree_child # number of tree children
        self.sigma = sigma # noise level

        # layers
        self.position_embeddings = nn.Embedding(self.context_length, self.n_embd) # position embeddings

        self._queries = nn.ModuleList() # query layers
        self._keys = nn.ModuleList() # key layers
        self._values = nn.ModuleList() # value layers
        self._mlps = nn.ModuleList() # MLP layers
        self._lns_1 = nn.ModuleList() # layer norm layers
        self._lns_2 = nn.ModuleList() # layer norm layers
        self.guided_layer_flag = [False] * n_layer # guided layer flag
        guided_layer_counter = 0 # guided layer counter
        for i in range(n_layer):
            self._queries.append(nn.Linear(n_embd, n_embd, bias=False))
            self._keys.append(nn.Linear(n_embd, n_embd, bias=False))
            self._values.append(nn.Linear(n_embd, n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd])) 
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(n_embd, n_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(n_mlp_hidden, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))

            if guide and guided_layer_counter<self.n_guided_layer*2 and (i+1) % self.guided_layer_gap == 0: # guided layer flag initialization
                self.guided_layer_flag[i] = True
                guided_layer_counter += 1

        self._read_out = nn.Linear(n_embd, 1) # read out layer
        self._out = nn.Linear(n_token, 1) #output layer 

    def token_embeddings(self, x):
        leave_options = torch.arange(0, self.vocab_size).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), self.vocab_size).to(x.device)
        leave_options = -torch.pow(leave_options - x.unsqueeze(-1),2)/2
        embeddings = torch.zeros(x.size(0), x.size(1), self.n_embd).to(x.device)
        embeddings[:,:,:self.vocab_size] = leave_options
        return embeddings

    def forward(self, x):
        contract_dim = 1 # contract dimension
        B, T = x.size() # batch size and position size
        positions = torch.arange(T, device=x.device).expand(B, T) # position embeddings
        H = self.token_embeddings(x) + self.position_embeddings(positions) # token embeddings + position embeddings 
        guided_counter = 0 # guided layer counter
        index_h = 0 # index of hd 
        index_q = self.n_guided_layer*self.vocab_size # index of qd
        index_u = 2*self.n_guided_layer*self.vocab_size # index of ud

        guided_layers = []
        for (q, k, v, mlp, ln1, ln2, guided_layer_flag) in zip(
            self._queries, self._keys, self._values, self._mlps, self._lns_1, self._lns_2, self.guided_layer_flag
        ):

            if self.layernorm: # layer norm
                H1 = ln1(H)
                query = q(H1) # query layer
                key = k(H1) # key layer
                value = v(H1) # value layer
            else:
                query = q(H) # query layer
                key = k(H) # key layer
                value = v(H) # value layer

            attn_weights = torch.einsum('bid,bjd->bij', query, key) # attention weights
            if self.normalize_attn: # normalize attention weights
                attn_weights = attn_weights / np.sqrt(H.shape[2]) 
            attn_weights = self.activation(attn_weights)
            H = H + torch.einsum('bij,bjd->bid', attn_weights, value) # attention layer + residual connection 
            

            if self.layernorm:
                H2 = ln2(H)
                if self.maxnorm: 
                    H2 = H2 - torch.max(H2, dim=2)[0].unsqueeze(2)
                if self.mlp: # MLP layer 
                    H = H + mlp(H2) 

            else: 
                if self.maxnorm:
                    H = H - torch.max(H, dim=2)[0].unsqueeze(2)
                if self.mlp: # MLP layer
                    H = H + mlp(H)
            
            if self.guide and guided_layer_flag: # guided layer
                
                if guided_counter < self.n_guided_layer: # downward process 
                    h_H = H[:,:, index_h: index_h+self.vocab_size] # hd 
                    q_H = H[:,:, index_q: index_q+self.vocab_size] # qd 
                    index_h += self.vocab_size # update index_h
                    index_q += self.vocab_size # update index_q
                    guide_output = torch.concatenate([h_H, q_H], dim=2) # concatenate hd and qd
                else:
                    index_h -= self.vocab_size # update index_h
                    index_q -= self.vocab_size # update index_q
                    h_H = H[:,:, index_h: index_h+self.vocab_size] # hd 
                    q_H = H[:,:, index_q: index_q+self.vocab_size] # qd 
                    u_H = H[:,:, index_u: index_u+self.vocab_size] # ud
                    index_u += self.vocab_size # update index_u
                    guide_output = torch.concatenate([h_H, q_H, u_H], dim=2) # concatenate hd, qd and ud
                guided_counter += 1
                guided_layers.append(
                    
                )
            
        prediction = self._read_out(H) # read out layer
        if self.guide:
            return prediction[:,:,0], guided_layers # return prediction and guided layers
        else:
            return prediction[:,:,0] # return prediction

class EncoderTransformer(nn.Module):
    def __init__(self, n_token, num_class, n_embd=128, n_layer=12, n_guided_layer=3, n_head=4, n_mlp_multiplier=4,
                 activation="softmax",  mlp=True, normalize_attn=True, layernorm=True, maxnorm=False, guide=False, guide_contract=False):
        super(EncoderTransformer, self).__init__()
        self.name = f"EncoderTF_embd={n_embd}_layer={n_layer}_head={n_head}"

        # Network structure configs
        # self.n_dims = n_dims # input dimension
        self.vocab_size = num_class # vocabulary size
        self.context_length = n_token # context length
        self.n_token = n_token # number of tokens
        self.n_embd = n_embd # embedding dimension
        self.n_head = n_head # number of heads
        self.n_layer = n_layer # number of layers
        self.n_mlp_hidden = n_embd * n_mlp_multiplier # number of hidden units in MLP
        self.activation = get_activation(activation) # activation function
        self.mlp = mlp # whether to use MLP

        # Normalization config
        self.normalize_attn = normalize_attn # whether to normalize attention
        self.layernorm = layernorm # whether to use layer normalization
        self.maxnorm = maxnorm # whether to use max normalization

        # Guided training config
        self.guide = guide # whether to use guided layer
        self.n_guided_layer =  n_guided_layer # number of guided layers
        self.guided_layer_gap = n_layer // n_guided_layer # guided layer gap 
        self.guide_contract = guide_contract


        # layers
        self.token_embeddings = nn.Embedding(self.vocab_size, self.n_embd) # token embeddings
        self.position_embeddings = nn.Embedding(self.context_length, self.n_embd) # position embeddings

        self._queries = nn.ModuleList() # query layers
        self._keys = nn.ModuleList() # key layers
        self._values = nn.ModuleList() # value layers
        self._mlps = nn.ModuleList() # MLP layers
        self._lns_1 = nn.ModuleList() # layer norm layers
        self._lns_2 = nn.ModuleList() # layer norm layers
        self.guided_layer_flag = [False] * n_layer # guided layer flag
        _layer_count = 0
        for i in range(n_layer):
            self._queries.append(nn.Linear(n_embd, n_embd, bias=False))
            self._keys.append(nn.Linear(n_embd, n_embd, bias=False))
            self._values.append(nn.Linear(n_embd, n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd])) 
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(n_embd, self.n_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(self.n_mlp_hidden, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))

            # guided layer flag initialization
            if guide and _layer_count<self.n_guided_layer and (i+1) % self.guided_layer_gap == 0: 
                self.guided_layer_flag[i] = True
                _layer_count += 1

        self._read_out = nn.Linear(n_embd, num_class) # read out layer
        self._out = nn.Linear(n_token, 1) #output layer 

    def forward(self, x):
        guide_contract_dim = 1 # guide_contract dimension
        B, T = x.size() # batch size and position size
        positions = torch.arange(T, device=x.device).expand(B, T) # position embeddings
        H = self.token_embeddings(x) + self.position_embeddings(positions) # token embeddings + position embeddings 
        guided_layers = []

        _layer_count = 0
        for (q, k, v, mlp, ln1, ln2, guided_layer_flag) in zip(
            self._queries, self._keys, self._values, self._mlps, self._lns_1, self._lns_2, self.guided_layer_flag
        ):
            H1 = ln1(H)
            query = q(H1) # query layer
            key = k(H1) # key layer
            value = v(H1) # value layer

            # attn_weights = self.activation(torch.einsum('bid,bjd->bij', query, key)) # attention weights
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) 
            if self.normalize_attn: # normalize attention weights
                attn_weights = attn_weights / np.sqrt(self.n_embd)
            attn_weights = self.activation(attn_weights)
            H = H + torch.einsum('bij,bjd->bid', attn_weights, value) # attention layer + residual connection 
            
            H2 = ln2(H)
            if self.maxnorm:
                H2 = H2 - torch.max(H2, dim=2)[0].unsqueeze(2)
            if self.mlp: # MLP layer
                H = H + mlp(H2)
            
            if self.guide and guided_layer_flag: # guided layer
                # upper_H = H[:,:,:self.vocab_size] # upper parts of H 
                if self.n_layer * self.vocab_size >= self.n_mlp_hidden: 
                    raise ValueError("The number of layers times the vocabulary size \
                                     should be less than the number of hidden units in MLP")
                upper_H = H[:,:,_layer_count*self.vocab_size:(_layer_count+1)*self.vocab_size] # rolling output block
                
                guided_layers.append(upper_H) # append guided layers
                # _layer_count += 1

        prediction = self._read_out(H) # read out layer
        # prediction = self.activation(prediction) # get class prediction 
        prediction = torch.transpose(prediction, 1, 2) # transpose prediction
        prediction = self._out(prediction)[:,:,0]  # output layer

  
        return prediction, guided_layers # return prediction and guided layers
       

class GuidedClassificationLoss(nn.Module):

    def __init__(self, penalty: float = 0.1):
        super(GuidedClassificationLoss, self).__init__()
        self.penalty = penalty # penalty weights

    def forward(self, inputs, targets):
        clas_input = inputs[0].view(-1, inputs[0].size(-1)) # classification input
        clas_target = targets[0].view(-1)
        loss = nn.functional.cross_entropy(clas_input, clas_target, reduction='none') # cross entropy loss
        guided_inputs = inputs[1]
        guided_targets = targets[1]
        for i in range(len(guided_inputs)):
            loss += self.penalty * torch.pow(torch.linalg.norm(guided_inputs[i] - guided_targets[i], dim=(1,2), ord='fro'), 2)
        return loss.mean() # return mean loss

class ClipLoss(nn.Module):
    def __init__(self, K, batch_size):
        super(ClipLoss, self).__init__()
        self.K = K 
        self.batch_size = batch_size 

    def forward(self, tmodel_output, imodel_output):
        # K text and 1 image 
        t_output_match = tmodel_output[:self.batch_size, :] 
        i_output_match = imodel_output[:self.batch_size, :]
        t_output_indep = tmodel_output[2*self.batch_size:, :] 

        # compute S 
        S_match = torch.exp(torch.sum( t_output_match * i_output_match, dim=1))
        S_indep = torch.exp(torch.sum( t_output_indep * torch.cat([i_output_match]*(self.K-1), dim=0), dim=1))
        concat_mat = torch.kron(torch.ones(self.K-1, 1), torch.eye(self.batch_size)).to(tmodel_output.device)
        sum_S_indep = torch.matmul(S_indep, concat_mat) 
        loss1 = -torch.log(S_match / (S_match + sum_S_indep))

        # K image and 1 text 
        t_output_match = tmodel_output[self.batch_size:2*self.batch_size, :]
        i_output_match = imodel_output[self.batch_size:2*self.batch_size, :]
        i_output_indep = imodel_output[2*self.batch_size:, :] 

        # compute S 
        S_match = torch.exp(torch.sum( t_output_match * i_output_match, dim=1))
        S_indep = torch.exp(torch.sum( i_output_indep * torch.cat([t_output_match]*(self.K-1), dim=0), dim=1))
        sum_S_indep = torch.matmul(S_indep, concat_mat)  
        loss2 = -torch.log(S_match / (S_match + sum_S_indep))

        return torch.mean(loss1 + loss2)

class GuidedClipLoss(nn.Module):
    def __init__(self, K, batch_size, penalty=1e-4, guide=False):
        super(GuidedClipLoss, self).__init__()
        self.K = K 
        self.batch_size = batch_size 
        self.penalty = penalty
        self.guide = guide

    def forward(self, tmodel_outputs, imodel_outputs, targets):

        tmodel_output = tmodel_outputs[0]
        imodel_output = imodel_outputs[0]
        # K text and 1 image 
        t_output_match = tmodel_output[:self.batch_size, :] 
        i_output_match = imodel_output[:self.batch_size, :]
        t_output_indep = tmodel_output[2*self.batch_size:, :] 

        # compute S 
        S_match = torch.exp(torch.sum( t_output_match * i_output_match, dim=1))
        S_indep = torch.exp(torch.sum( t_output_indep * torch.cat([i_output_match]*(self.K-1), dim=0), dim=1))
        concat_mat = torch.kron(torch.ones(self.K-1, 1), torch.eye(self.batch_size)).to(tmodel_output.device)
        sum_S_indep = torch.matmul(S_indep, concat_mat) 
        loss1 = -torch.log(S_match / (S_match + sum_S_indep))

        # K image and 1 text 
        t_output_match = tmodel_output[self.batch_size:2*self.batch_size, :]
        i_output_match = imodel_output[self.batch_size:2*self.batch_size, :]
        i_output_indep = imodel_output[2*self.batch_size:, :] 

        # compute S 
        S_match = torch.exp(torch.sum( t_output_match * i_output_match, dim=1))
        S_indep = torch.exp(torch.sum( i_output_indep * torch.cat([t_output_match]*(self.K-1), dim=0), dim=1))
        sum_S_indep = torch.matmul(S_indep, concat_mat)  
        loss2 = -torch.log(S_match / (S_match + sum_S_indep))

        loss = loss1 + loss2 
        loss = loss.mean()

        loss3 = 0
        if self.guide:
            tguided_input = tmodel_outputs[1]
            iguided_input = imodel_outputs[1]
            tguided_target = targets[0]
            iguided_target = targets[1]
            for i in range(len(tguided_input)):
                loss3 += self.penalty * torch.pow(torch.linalg.norm(tguided_input[i] - tguided_target[i], dim=(1,2), ord='fro'), 2)
            
            for i in range(len(iguided_input)):
                loss3 += self.penalty * torch.pow(torch.linalg.norm(iguided_input[i] - iguided_target[i], dim=(1,2), ord='fro'), 2)
            
            loss += loss3.mean()
            loss3 = loss3.mean().item()/self.penalty

        return loss, loss3 



class SoftmaxClipLoss(nn.Module):

    def __init__(self, K, batch_size):
        super(SoftmaxClipLoss, self).__init__()
        self.K = K 
        self.batch_size = batch_size 
    
    def forward(self, tmodel_output, imodel_output):

        # softmax 
        tmodel_output = F.softmax(tmodel_output, dim=1) 
        imodel_output = F.softmax(imodel_output, dim=1)

        # K text and 1 image 
        t_output_match = tmodel_output[:self.batch_size, :] 
        i_output_match = imodel_output[:self.batch_size, :]
        t_output_indep = tmodel_output[2*self.batch_size:, :] 
        concat_mat = torch.kron(torch.ones(self.K-1, 1), torch.eye(self.batch_size)).to(tmodel_output.device)
        # compute S 
        S_match = torch.sum( t_output_match * i_output_match, dim=1)
        S_indep = torch.sum( t_output_indep * torch.cat([i_output_match]*(self.K-1), dim=0), dim=1) 
        sum_S_indep = torch.matmul(S_indep, concat_mat) 
        loss1 = -torch.log(S_match / (S_match + sum_S_indep))

        # K image and 1 text 
        t_output_match = tmodel_output[self.batch_size:2*self.batch_size, :]
        i_output_match = imodel_output[self.batch_size:2*self.batch_size, :]
        i_output_indep = imodel_output[2*self.batch_size:, :] 

        # compute S 
        S_match = torch.sum( t_output_match * i_output_match, dim=1)
        S_indep = torch.sum( i_output_indep * torch.cat([t_output_match]*(self.K-1), dim=0), dim=1)
        sum_S_indep = torch.matmul(S_indep,concat_mat) 
        loss2 = -torch.log(S_match / (S_match + sum_S_indep))

        return torch.mean(loss1 + loss2)

class GuidedLsLoss(nn.Module):
    def __init__(self,  penalty=1e-4):
        super(GuidedLsLoss, self).__init__()
        self.penalty = penalty

    def forward(self, inputs, targets):
        loss = torch.sum(torch.pow(inputs[0] - targets[0], 2), dim=1)
        guided_input = inputs[1]
        guided_target = targets[1]
        loss3 = 0
        for i in range(len(guided_input)):
            loss3 += self.penalty * torch.pow(torch.linalg.norm(guided_input[i] - guided_target[i], dim=(1,2), ord='fro'), 2)
    
        return loss.mean() +loss3.mean()
    
class ConditionalGuidedLsLoss(nn.Module):
    def __init__(self,  penalty=1e-4, guide=False):
        super(ConditionalGuidedLsLoss, self).__init__()
        self.penalty = penalty
        self.guide = guide

    def forward(self, inputs, targets, verbose=False):
        loss = torch.sum(torch.pow(inputs[0] - targets[0], 2), dim=1)
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss5 = 0

        if verbose:
            guided_input = inputs[1][0]
            guided_target = targets[1][0]
            print("Text guided info")
            for i in range(len(guided_input)):
                print("Layer", i)
                print(guided_input[i].shape)
                print(guided_target[i].shape)
            
            print("Image guided info")
            guided_input = inputs[1][1]
            guided_target = targets[1][1]
            for i in range(len(guided_input)):
                print("Layer", i)
                print(guided_input[i].shape)
                print(guided_target[i].shape)


        if self.guide:
            guided_input = inputs[1][1]
            guided_target = targets[1][1]
            for i in range(len(guided_input)//2):
                loss2 += self.penalty * torch.pow(torch.linalg.norm(guided_input[i] - guided_target[i], dim=(1,2), ord='fro'), 2)
                loss5 += self.penalty *torch.pow(torch.linalg.norm(guided_input[i+len(guided_input)//2 +1] - guided_target[i+len(guided_input)//2+1], dim=(1,2), ord='fro'), 2)
            loss4 += self.penalty *torch.pow(torch.linalg.norm(guided_input[len(guided_input)//2 ] - guided_target[len(guided_input)//2], dim=(1,2), ord='fro'), 2)

            guided_input = inputs[1][0]
            guided_target = targets[1][0]
            for i in range(len(guided_input)):
                loss3 += self.penalty * torch.pow(torch.linalg.norm(guided_input[i] - guided_target[i], dim=(1,2), ord='fro'), 2)
            loss = loss + loss2 + loss3+ loss4 + loss5
    
            return loss.mean(), loss2.mean().item(),loss4.mean().item(), loss5.mean().item(), loss3.mean().item()
        else:
            return loss.mean(), 0, 0, 0, 0
        


class GuidedCELoss(nn.Module):
    def __init__(self,  penaltys, guide=False):
        super().__init__()
        self.penaltys = penaltys
        self.guide = guide

    def forward(self, inputs, targets):
        loss = self.penaltys[0]*nn.functional.cross_entropy(inputs[0], targets[0], reduction='none') # cross entropy loss

        if self.guide:
            loss2 = 0
            guided_input = inputs[1]
            guided_target = targets[1]
            for i in range(len(guided_input)):
                loss2 += self.penaltys[1] * torch.pow(torch.linalg.norm(guided_input[i] - guided_target[i], dim=(1,2), ord='fro'), 2)
            loss = loss + loss2
        
        return loss.mean()

class KLdiv(nn.Module):
    def __init__(self):
        super(KLdiv, self).__init__()
    
    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1, inputs.size(-1))
        targets = targets.reshape(-1, targets.size(-1))
        inputs = F.log_softmax(inputs, dim=1)
        return nn.functional.kl_div(inputs, targets, reduction='batchmean')

class ConditionalGuidedCELoss(nn.Module):
    def __init__(self,  penalty=1e-4, guide=False):
        super().__init__()
        self.penalty = penalty
        self.guide = guide

    def forward(self, inputs, targets, verbose=False):
        clas_inputs = inputs[0].reshape(-1, inputs[0].size(-1)) # classification input
        # print(clas_inputs.shape)
        clas_targets = targets[0].reshape(-1)
        # print(clas_targets.shape)
        loss = nn.functional.cross_entropy(clas_inputs, clas_targets, reduction='none') # cross entropy loss
        # print(loss.shape)
        # print("target",targets[0].shape)
        loss = loss.reshape(-1, targets[0].shape[1])
        loss = torch.mean(loss, dim=1)
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss5 = 0

        if verbose:
            guided_input = inputs[1][0]
            guided_target = targets[1][0]
            print("Text guided info")
            for i in range(len(guided_input)):
                print("Layer", i)
                print(guided_input[i].shape)
                print(guided_target[i].shape)
            
            print("Image guided info")
            guided_input = inputs[1][1]
            guided_target = targets[1][1]
            for i in range(len(guided_input)):
                print("Layer", i)
                print(guided_input[i].shape)
                print(guided_target[i].shape)


        if self.guide:
            guided_input = inputs[1][0]
            guided_target = targets[1][0]
            # print(guided_input[0].shape)
            # print(guided_target[0].shape)
            for i in range(len(guided_input)//2):
                loss2 += self.penalty * torch.pow(torch.linalg.norm(guided_input[i] - guided_target[i], dim=(1,2), ord='fro'), 2)
                loss5 += self.penalty *torch.pow(torch.linalg.norm(guided_input[i+len(guided_input)//2 +1] - guided_target[i+len(guided_input)//2+1], dim=(1,2), ord='fro'), 2)
            loss4 += self.penalty *torch.pow(torch.linalg.norm(guided_input[len(guided_input)//2 ] - guided_target[len(guided_input)//2], dim=(1,2), ord='fro'), 2)

            guided_input = inputs[1][1]
            guided_target = targets[1][1]
            for i in range(len(guided_input)):
                loss3 += self.penalty * torch.pow(torch.linalg.norm(guided_input[i] - guided_target[i], dim=(1,2), ord='fro'), 2)
            loss = loss + loss2 + loss3+ loss4 + loss5
            # loss = loss2 + loss3+ loss4 + loss5
            # loss = loss3 


            loss2 = loss2.mean().item()
            loss3 = loss3.mean().item()
            loss4 = loss4.mean().item()
            loss5 = loss5.mean().item()
            return loss.mean(), loss2, loss4, loss5,loss3
        else:
            return loss.mean(), 0, 0, 0, 0
        

class LsLoss(nn.Module):
    def __init__(self):
        super(LsLoss, self).__init__()
    
    def forward(self, inputs, targets):
        return torch.sum(torch.pow(inputs - targets, 2), dim=1).mean()
    
class GuidedSoftmaxClipLoss(nn.Module):

    def __init__(self, K, batch_size, penalty=1e-4):
        super(GuidedSoftmaxClipLoss, self).__init__()
        self.K = K 
        self.batch_size = batch_size 
        self.penalty = penalty
    
    def forward(self, tmodel_outputs, imodel_outputs, targets):

        tmodel_output = tmodel_outputs[0]
        imodel_output = imodel_outputs[0]

        # softmax 
        tmodel_output = F.softmax(tmodel_output, dim=1) 
        imodel_output = F.softmax(imodel_output, dim=1)

        # K text and 1 image 
        t_output_match = tmodel_output[:self.batch_size, :] 
        i_output_match = imodel_output[:self.batch_size, :]
        t_output_indep = tmodel_output[2*self.batch_size:, :] 
        concat_mat = torch.kron(torch.ones(self.K-1, 1), torch.eye(self.batch_size)).to(tmodel_output.device)
        # compute S 
        S_match = torch.sum( t_output_match * i_output_match, dim=1)
        S_indep = torch.sum( t_output_indep * torch.cat([i_output_match]*(self.K-1), dim=0), dim=1) 
        sum_S_indep = torch.matmul(S_indep, concat_mat) 
        loss1 = -torch.log(S_match / (S_match + sum_S_indep))

        # K image and 1 text 
        t_output_match = tmodel_output[self.batch_size:2*self.batch_size, :]
        i_output_match = imodel_output[self.batch_size:2*self.batch_size, :]
        i_output_indep = imodel_output[2*self.batch_size:, :] 

        # compute S 
        S_match = torch.sum( t_output_match * i_output_match, dim=1)
        S_indep = torch.sum( i_output_indep * torch.cat([t_output_match]*(self.K-1), dim=0), dim=1)
        sum_S_indep = torch.matmul(S_indep,concat_mat) 
        loss2 = -torch.log(S_match / (S_match + sum_S_indep))

        loss = loss1 + loss2 
        tguided_input = tmodel_outputs[1]
        iguided_input = imodel_outputs[1]
        tguided_target = targets[0]
        iguided_target = targets[1]

        loss3 = 0
        for i in range(len(tguided_input)):
            loss3 += self.penalty * torch.pow(torch.linalg.norm(tguided_input[i] - tguided_target[i], dim=(1,2), ord='fro'), 2)
        
        for i in range(len(iguided_input)):
            loss3 += self.penalty * torch.pow(torch.linalg.norm(iguided_input[i] - iguided_target[i], dim=(1,2), ord='fro'), 2)

        return loss.mean() +loss3.mean()

if __name__ == "__main__":
    # model = DenoiseEncoderTransformer(n_token=27, num_class=10, 
    #                                 n_embd=128, n_layer=12, n_guided_layer=3,
    #                                 n_head=4, n_mlp_hidden=512, 
    #                                 activation="relu", 
    #                                 mlp=True, normalize_attn=True, 
    #                                 layernorm=True, maxnorm=False, 
    #                                 guide=True, sigma=1)

    # z = torch.randint(0, 10, (10, 27))
    # y, guided_layers = model(z)
    # for guided_layer in guided_layers:
    #     # print(guided_layer)
    #     print(guided_layer.shape)

    # model = ConditionalDenoiseEncoderTransformer(n_token=20, 
    #                                              n_i_token=10,
    #                                              num_class=10,
    #                                                 n_embd=128,
    #                                                 n_layer=12,
    #                                                 n_guided_layers=[3, 4],
    #                                                 n_head=4,
    #                                                 n_mlp_hidden=512,
    #                                                 activation="relu",
    #                                                 mlp=True,
    #                                                 normalize_attn=True,
    #                                                 layernorm=True,
    #                                                 maxnorm=False,
    #                                                 guide=True,
    #                                                 sigma=1)
    # z= torch.randint(0, 10, (10, 20))
    # y, guided_layers = model(z)
    # print("t_guided_layers")
    # for guided_layer in guided_layers[0]:
    #     print(guided_layer.shape)
    
    # print("i_guided_layers")
    # for guided_layer in guided_layers[1]:
    #     print(guided_layer.shape)

    n_token = 8
    n_i_token = 3
    print(generate_mask(n_token, n_i_token, 1))

    model = AutoRegressiveTransformer(guide=True) 
    xt = torch.randint(0, 10, (10, 5))
    zi = torch.randint(0, 10, (10, 4))
    y, guided_layers = model(xt, zi)
    for guided_layer in guided_layers[0]:
        print(guided_layer.shape)
    for guided_layer in guided_layers[1]:
        print(guided_layer.shape)