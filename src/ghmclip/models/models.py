###############################################################################
# This file has two parts. 
# 1. Commonly used architectures for transformer models.
# 2. Specialized architectures for different tasks.
###############################################################################

from re import S
import torch
import torch.nn as nn
import copy
import math
import random
import os
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

###############################################################################
# Part 1. Commonly Used Architectures
###############################################################################

def attention(query, key, value, mask=None):
    "Compute Scaled Dot Product Attention" 
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-1,-2))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = scores.softmax(dim=-1)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
	def __init__(self, h, d_model):
		"Take in model size and number of heads."
		super(MultiHeadAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None

	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Add batch dimension and head dimension
			mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
		nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
		query, key, value = [
			lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			for lin, x in zip(self.linears, (query, key, value))
		]

		# 2) Apply attention on all the projected vectors in batch.
		x, self.attn = attention(
			query, key, value, mask=mask
		)

		# 3) "Concat" using a view and apply a final linear.
		x = (
			x.transpose(1, 2)
			.contiguous()
			.view(nbatches, -1, self.h * self.d_k)
		)
		del query
		del key
		del value
		return self.linears[-1](x)

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
        return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = RMSNorm(size)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + sublayer(self.norm(x))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = RMSNorm(layer.size) 

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


###############################################################################
# Part 2. Specialized Architectures
###############################################################################

class EncoderTransformer(Encoder):
    """
    Encoder only transformer model.
    This model implements a transformer architecture with only the encoder component.
    It supports guided layers for extracting intermediate representations, auto-regressive
    mode for sequential prediction, and feature extraction for downstream tasks.
    
    Attributes:
        guide_flag (list): Flags indicating which layers are guided layers
        n_class (int): Number of output classes
        auto_regressive (bool): Whether to use auto-regressive masking
        token_embeddings (nn.Embedding): Embedding layer for input tokens
        position_embeddings (nn.Embedding): Embedding layer for positional encoding
        readout (nn.Linear): Linear layer for final classification
        out (nn.Linear): Linear layer for sequence-level output
        rep (bool): Whether to return intermediate representations
        guide_dim (list): Dimensions of guided layers
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
        feature (bool, optional): Whether to return intermediate features. Defaults to False.
    
    Returns:
        tuple: A tuple containing:
            - prediction (torch.Tensor): Output logits of shape (batch_size, n_class)
            - guided_info (list): List of guided layer outputs if guide=True
            - features (list): List of intermediate features if feature=True
    """
    def __init__(self, 
                 n_head=1, 
                 d_model=128, 
                 d_ff=512, 
                 n_layer=6, 
                 guide=False, 
                 n_class=10, 
                 n_token=81, 
                 auto_regressive=False, 
                 job='cls',
                 guide_dim=None):
        
        # Initialize the encoder layers
        layer = EncoderLayer(d_model, MultiHeadAttention(n_head, d_model), FFN(d_model, d_ff))
        # Copy the encoder layers n_layer times
        super(EncoderTransformer, self).__init__(layer, n_layer)

        self.guide_flag = [False] * n_layer # guide flag
        self.n_class = n_class # number of output classes
        self.auto_regressive = auto_regressive # auto-regressive mode
        self.token_embeddings = nn.Embedding(n_class, d_model) # token embeddings
        self.d_model = d_model # model dimension
        self.mask = None # mask
        self.position_embeddings = nn.Embedding(n_token, d_model) # position embeddings
        self.job = job # job type
        if job=='dns':
            self.readout = nn.Linear(d_model, 1) # readout layer 
        else:
            self.readout = nn.Linear(d_model, n_class) # readout layer 
        self.out = nn.Linear(n_token , 1) # sequence-level output layer

        self.guide_dim = guide_dim # dimensions of guided layers
        if guide: # if guide is True
            n_guided_layer = len(guide_dim) # number of guided layers
            assert n_guided_layer <= n_layer, "n_guided_layer must be less than or equal to n_layer"
            gap = n_layer // n_guided_layer # gap between guided layers
            for i in range(n_guided_layer):
                assert torch.max(guide_dim[i]) <= d_model, "guide_dim must be less than or equal to d_model"
                self.guide_flag[(i+1)*gap-1] = True # set the guide flag to True

    def forward(self, x, feature=False):
        
        B, T = x.size() # batch size and position size
        positions = torch.arange(T, device=x.device).expand(B, T) # position embeddings
        if self.job == 'dns':
            leave_options = torch.arange(0, self.n_class, device=x.device).view(1, 1, -1)
            leave_options = -torch.pow(leave_options - x.unsqueeze(-1), 2) / 2
            embeddings = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)
            embeddings[:,:,:self.n_class] = leave_options
            x = embeddings + self.position_embeddings(positions) # token embeddings + position embeddings 
        else:
             x = self.token_embeddings(x) + self.position_embeddings(positions) # token embeddings + position embeddings
            # x = self.token_embeddings(x)# position embeddings

        guided_info = [] # guided information
        features = [x] # features
        _layer_count = 0

        # Generate mask for auto-regressive model
        if self.auto_regressive and self.mask is None:
            # Create lower triangular mask (including diagonal)
            self.mask = torch.tril(torch.ones(x.size(1), x.size(1))).bool().to(x.device)

        # Forward pass
        for i, layer in enumerate(self.layers):
            x = layer(x, self.mask)

            if feature:
                features.append(x)
            
            if self.guide_flag[i]:
                upper_x = x[:,:,self.guide_dim[_layer_count]] # guided block
                guided_info.append(upper_x) # collect guided information 
                _layer_count += 1
        
        x = self.readout(self.norm(x)) # (B, T, n_class)
        if self.job == 'cls' or self.job == 'clip':
            x = torch.transpose(x, 1, 2) # (B, n_class, T)
            x = self.out(x)[:,:,0] # (B, n_class)
        elif self.job== 'dns':
            x = x[:,:,0] # (B, T)
        elif self.job == 'nwp':
            x = torch.transpose(x, 1, 2) # (B, n_class, T)
        
        if feature:
            features.append(x)
        
        return x, guided_info, features

    
    
    