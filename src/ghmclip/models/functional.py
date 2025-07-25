###############################################################################
# This file has two parts. 
# 1. Those guided losses for our models.
# 2. Useful functions for training and testing.
###############################################################################
import torch 
import torch.nn as nn
import torch.nn.functional as F



############################################################################### 
# Part 1. Guided Losses
############################################################################### 

class MSELoss(nn.Module):
    def __init__(self,  penalty=0):
        super(MSELoss, self).__init__()
        self.penalty = penalty

    def forward(self, inputs, targets):
        loss = torch.sum(torch.pow(inputs[0] - targets[0], 2), dim=1)
        guided_input = inputs[1]
        guided_target = targets[1]
        guided_term = torch.zeros(inputs[0].shape[0], device=loss.device)

        if self.penalty !=0:
            for i in range(len(guided_input)):
                guided_term += self.penalty * torch.pow(torch.linalg.norm(guided_input[i] - guided_target[i], dim=(1,2), ord='fro'), 2)
            total_loss = loss + guided_term
    
        return total_loss.mean(), loss.mean().item(), guided_term.mean().item()

class CELoss(nn.Module):
    def __init__(self,  penalty=0):
        super().__init__()
        self.penalty = penalty

    def forward(self, inputs, targets):
        loss = nn.functional.cross_entropy(inputs[0], targets[0], reduction='none') # cross entropy loss
        loss = loss.mean(dim=1)
        guided_term = torch.zeros(inputs[0].shape[0], device=loss.device)

        if self.penalty !=0:
            guided_input = inputs[1]
            guided_target = targets[1]
            for i in range(len(guided_input)):
                guided_term += self.penalty * torch.pow(torch.linalg.norm(guided_input[i] - guided_target[i], dim=(1,2), ord='fro'), 2)
            total_loss = loss + guided_term
        
        return total_loss.mean(), loss.mean().item(), guided_term.mean().item()

class CLIPLoss(nn.Module):
    def __init__(self, penalty=0, normalize=True, temperature=1.0):
        super().__init__()
        self.penalty = penalty
        self.normalize = normalize
        self.temperature = temperature

    def forward(self, inputs, targets, guide=False):
        text_input = inputs[0]
        image_input = inputs[1]
        text_embed = text_input[0]
        image_embed = image_input[0]

        if self.normalize:
            text_embed = F.normalize(text_embed, p=2, dim=1)
            image_embed = F.normalize(image_embed, p=2, dim=1)
        # print("text_embed.shape, image_embed.shape: ", text_embed.shape, image_embed.shape)
        Similarity = torch.exp(text_embed@(image_embed.T)/self.temperature) 
        loss_text_image = torch.diag(Similarity)/torch.sum(Similarity, dim=1) 
        loss_image_text = torch.diag(Similarity)/torch.sum(Similarity, dim=0) 
        total_loss = -torch.log(loss_text_image) - torch.log(loss_image_text) 
        total_loss = total_loss.mean()
        loss = total_loss.item()

        if self.penalty != 0 and guide:
            text_guided_input = text_input[1]
            image_guided_input = image_input[1]
            text_guided_target = targets[0]
            image_guided_target = targets[1]
            
            # Vectorize guided loss computation
            text_guided_diff = torch.stack(text_guided_input) - torch.stack(text_guided_target)
            image_guided_diff = torch.stack(image_guided_input) - torch.stack(image_guided_target)
            
            penality_term = self.penalty * (
                torch.sum(torch.pow(torch.linalg.norm(text_guided_diff, dim=(2,3), ord='fro'), 2),dim=0) +
                torch.sum(torch.pow(torch.linalg.norm(image_guided_diff, dim=(2,3), ord='fro'), 2),dim=0)
            )
            
            total_loss += penality_term.mean()
            penality_term = penality_term.mean().item()/self.penalty
        else:
            penality_term = 0
        
        return total_loss, loss, penality_term

        
            


############################################################################### 
# Part 2. Useful functions
###############################################################################

def gen_guide_indices(job='dns', n_tree_layer=10, n_class=10):
    """
    Generate the indices of the guided layers for the given job name and number of tree layers.

    Args:
        job (str): The name of the job. "clas" or "dns" or "nwp" or "clip"
        n_tree_layer (int): The number of tree layers.

    Returns:
        list: The indices of the guided layers.
    """
    indices = []
    if job == 'clas' or job == 'clip':
        for i in range(n_tree_layer):
            indices.append(torch.arange(i*n_class, (i+1)*n_class))
    elif job == 'dns':
        index_h = 0
        index_q = (n_tree_layer+1)*n_class 
        index_u = 2*(n_tree_layer+1)*n_class  

        for i in range(n_tree_layer+1):
            indice1 = torch.arange(index_h, index_h+n_class) 
            indice2 = torch.arange(index_q, index_q+n_class) 
            indice = torch.cat([indice1, indice2], dim=0) 
            indices.append(indice)
            index_h += n_class
            index_q += n_class
        
        for i in range(n_tree_layer):
            index_h -= n_class
            index_q -= n_class 
            indice1 = torch.arange(index_h, index_h+n_class)
            indice2 = torch.arange(index_q, index_q+n_class)
            indice3 = torch.arange(index_u, index_u+n_class) 
            indice = torch.cat([indice1, indice2, indice3], dim=0)
            indices.append(indice)
            index_u += n_class
    
    elif job == 'nwp':
        index_q = 0 # index of hd 
        index_h = (n_tree_layer+1)*n_class # index of qd
        index_u = (2*n_tree_layer+1)*n_class # index of ud

        indices.append(torch.arange(index_q, index_q+n_class)) 
        index_q += n_class

        for i in range(n_tree_layer):
            indice1 = torch.arange(index_h, index_h+n_class) 
            indice2 = torch.arange(index_q, index_q+n_class) 
            indice = torch.cat([indice1, indice2], dim=0) 
            indices.append(indice)
            index_h += n_class
            index_q += n_class
        
        for i in range(n_tree_layer):
            indices.append(torch.arange(index_u, index_u+n_class))
            index_u = index_u + n_class 

    return indices

