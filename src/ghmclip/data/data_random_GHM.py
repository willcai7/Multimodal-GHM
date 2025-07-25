"""
Code for generating random data using the Gnerative Hierarchical Model (GHM). 
"""

# Modules
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Functions 

def GenTransition(n_layer, 
                  n_child, 
                  variable_type, 
                  p_flip = 0.3, 
                  flip_scale = 1.0, 
                  translation_invariance=True,
                  verbose=False,
                  device='cpu',
                  seedtree=42):
    """
    Generate the transition matrix for the GHM tree. 

    Args:
    n_layer: int, number of layers in the tree
    n_child: int, number of children for each node
    variable_type: int, number of possible values for each node
    p_flip: float, the probability of flipping the transition matrix
    flip_scale: float, the scale of the Gaussian distribution for flipping the transition matrix
    translation_invariance: bool, whether the transition matrix is translation invariant

    Returns:
    transition: list of list of numpy array, the transition matrix for each layer
    """
    torch.manual_seed(seedtree)
    transition = [[] for _ in range(n_layer)]
    if verbose:
        skeleton = []
    if translation_invariance: 
        for layer in range(n_layer):
            # Create identity matrix and permute rows
            identity = torch.eye(variable_type, device=device)
            perm_indices = torch.randperm(variable_type, device=device)
            skeleton_matrix = identity[perm_indices, :]
            
            # Create transitions for each child
            invariance_transition = []
            for _ in range(n_child):
                # Generate random normal tensor
                random_tensor = torch.normal(0, flip_scale, (variable_type, variable_type), device=device)
                # Apply softmax row-wise
                softmax_tensor = torch.softmax(random_tensor, dim=1)
                # Combine with skeleton matrix
                transition_matrix = (1 - p_flip) * skeleton_matrix + p_flip * softmax_tensor
                invariance_transition.append(transition_matrix)
            
            # Extend transitions for this layer
            for _ in range(n_child ** layer):            
                transition[layer].extend(invariance_transition)
            
            if verbose:
                skeleton.append(skeleton_matrix)
    else:
        for layer in range(n_layer):
            for _ in range(n_child ** layer):
                layer_transitions = []
                for _ in range(n_child):
                    # Create identity matrix and permute rows
                    identity = torch.eye(variable_type, device=device)
                    perm_indices = torch.randperm(variable_type, device=device)
                    skeleton_matrix = identity[perm_indices, :]
                    
                    # Generate random normal tensor
                    random_tensor = torch.normal(0, flip_scale, (variable_type, variable_type), device=device)
                    # Apply softmax row-wise
                    softmax_tensor = torch.softmax(random_tensor, dim=1)
                    # Combine with skeleton matrix
                    transition_matrix = (1 - p_flip) * skeleton_matrix + p_flip * softmax_tensor
                    layer_transitions.append(transition_matrix)
                
                transition[layer].extend(layer_transitions)
    
    if verbose:
        return transition, skeleton
    else:
        return transition

# Basic Classes 

class Node:
    """
    Node class for the GHM tree.
    """
    def __init__(self, value = None, 
                 parent = None, 
                 children = None):
        self.value = value
        self.parent = parent
        self.children = children
        self.hd_message = 0

class GHMTree:
    """
    GHM tree class using PyTorch tensors.
    """
    def __init__(self, 
                 n_layer=4, 
                 n_child=3, 
                 variable_type=10, 
                 p_y=None,  # Will convert to tensor
                 p_flip=0.3, 
                 transition=None, 
                 batch_size=128, 
                 build_tree=False, 
                 root=None,
                 device='cpu'):
        
        self.device = device
        self.variable_type = variable_type
        self.posterior_probability_CLS = None
        self.n_layer = n_layer
        self.n_child = n_child
        # Convert p_y to tensor if provided
        if p_y is None:
            self.p_y = torch.ones(variable_type, device=device) / variable_type
        else:
            self.p_y = p_y.clone().detach().to(device)
        self.p_flip = p_flip
        # vert transition matrices to tensors
        if transition is None:
            self.transition = GenTransition(n_layer, n_child, variable_type, p_flip, flip_scale, translation_invariance, verbose, device)
        else:
            self.transition = transition
        self.batch_size = batch_size
        self.root = root
        self.build_tree_flag = build_tree
        self.dns_flag = False 
        self.cls_flag = False 

        self.gen_values()
        if self.build_tree_flag:
            self.build_tree()

    def gen_values(self):
        """
        Generate values for the tree using PyTorch.
        """
        self.T_value = [[] for _ in range(self.n_layer + 1)]
        if self.root is not None:
            self.T_value[0].append(self.root)
        else:
            # Use torch.multinomial for sampling
            # Sample from p_y distribution for batch_size samples
            # p_y needs to be properly shaped for multinomial sampling
            sampled_values = torch.multinomial(self.p_y, num_samples=self.batch_size,replacement=True).squeeze()
            self.T_value[0].append(sampled_values)

        for layer in range(1, self.n_layer + 1):
            for id in range(len(self.T_value[layer - 1])):
                node_value = self.T_value[layer - 1][id]
                for id_child in range(self.n_child):
                    # Get transition probabilities for the current node and child
                    trans_prob = self.transition[layer-1][id * self.n_child + id_child][node_value]
                    
                    # Sample values using multinomial distribution (much cleaner approach)
                    sampled_values = torch.multinomial(trans_prob, num_samples=1, replacement=True).squeeze(1)
                    
                    # Store the sampled values for this node's child
                    self.T_value[layer].append(sampled_values)

    def build_tree(self):
        """
        Build the tree and store the values into the nodes. 
        """
        self.posterior_probability_CLS = None 
        self.posterior_mean_DNS = None 
        self.Tree = [[] for _ in range(self.n_layer+1)]
        self.Tree[0] = [Node(self.T_value[0][0])]
        self.Tree[0][0].children = list(Node() for _ in range(self.n_child))
        for layer in range(1, self.n_layer+1):
            index = 0
            for node in self.Tree[layer - 1]:
                node.children = list(Node(self.T_value[layer][index+i], parent = node) for i in range(self.n_child))
                index += self.n_child
                self.Tree[layer].extend(node.children)
    
    def BP_CLS(self):
        """
        Belief propagation for classification using PyTorch.
        """
        # Generate message for the L-1 layer nodes
        for id_node in range(len(self.Tree[-2])):
            node = self.Tree[-2][id_node]
            node.hd_message = torch.zeros((self.variable_type, self.batch_size), device=self.device)
            for id_child in range(self.n_child):
                leaf_value = self.T_value[-1][id_node*self.n_child+id_child]
                trans_prob = self.transition[-1][id_node * self.n_child + id_child]
                node.hd_message += torch.log(trans_prob[:, leaf_value])
            node.hd_message -= torch.max(node.hd_message, dim=0)[0]

        # Generate message for layer L-2 to 0
        for layer in range(self.n_layer-2, -1, -1):
            for id_node in range(len(self.Tree[layer])):
                node = self.Tree[layer][id_node]
                node.hd_message = torch.zeros((self.variable_type, self.batch_size), device=self.device)
                for id_child in range(self.n_child):
                    child = node.children[id_child]
                    trans_prob = self.transition[layer][id_node * self.n_child + id_child]
                    node.hd_message += torch.log(torch.mm(trans_prob, torch.exp(child.hd_message)))
                node.hd_message -= torch.max(node.hd_message, dim=0)[0]

        # Final probability
        h0 = self.root_node.hd_message + torch.log(self.p_y).reshape(-1, 1)
        h0 -= torch.max(h0, dim=0)[0]
        self.posterior_probability_CLS = torch.exp(h0) / torch.sum(torch.exp(h0), dim=0)
        self.posterior_probability_CLS = self.posterior_probability_CLS.T

        self.cls_flag = True

        return self.posterior_probability_CLS    
    
    def BP_dummy_NWP(self, position, external_hd_message=None):
        """
        Belief propagation for denoising tasks. 
        """
        # Generate message for the leaf nodes 
        vt_choices = torch.linspace(0, self.variable_type - 1, self.variable_type, device=self.device)
        index = 0
        for id_node in range(len(self.Tree[-1])):
            node = self.Tree[-1][id_node]
            if id_node >= position:
                node.hd_message = torch.zeros((self.variable_type, 1), device=self.device)
                node.qd_message = torch.log(torch.mm(self.transition[-1][id_node], torch.exp(node.hd_message)))
            else:
                node.hd_message = None
                node.qd_message = torch.log(self.transition[-1][id_node][:, self.T_value[-1][id_node]])

        # Downward process 
        for layer in range(self.n_layer-1, 0, -1):
            for id_node in range(len(self.Tree[layer])):
                node = self.Tree[layer][id_node]
                node.hd_message = sum(child.qd_message for child in node.children)
                node.qd_message = torch.log(torch.mm(self.transition[layer-1][id_node], torch.exp(node.hd_message)))

        # Update the root node 
        self.root_node.hd_message = sum(child.qd_message for child in self.root_node.children)
        self.root_node.bu_message = self.root_node.hd_message

        if external_hd_message is not None:
            self.root_node.bu_message += external_hd_message

        # Upward process
        for layer in range(1, self.n_layer+1):
            for id_node in range(len(self.Tree[layer])):
                node = self.Tree[layer][id_node]
                diff = node.parent.bu_message - node.qd_message
                if node.hd_message is not None:
                    node.bu_message = node.hd_message + torch.log(torch.mm(self.transition[layer-1][id_node].T, torch.exp(diff)))

        return torch.exp(self.Tree[-1][position].bu_message)/torch.sum(torch.exp(self.Tree[-1][position].bu_message), 0)

    def BP_NWP(self, position, external_hd_message=None):
        """
        Belief propagation for the next word prediction task. 
        """
        current_layer_position = position-1
        object_poisiton = position 
        current_object_layer_position = object_poisiton
        share_parents_flags = [False]
        object_parents = [position]
        
        for id_node in range(position):
            node = self.Tree[-1][id_node]
            node.qd_message = torch.log(self.transition[-1][id_node][:, self.T_value[-1][id_node]])

        # Downward process 
        for layer in range(self.n_layer-1, 0, -1):
            previous_layer_position = current_layer_position
            current_layer_position = (current_layer_position) // self.n_child 
            current_object_layer_position = (current_object_layer_position) // self.n_child
            share_parents_flags.append(current_layer_position == current_object_layer_position)
            object_parents.append(current_object_layer_position)
            
            for id_node in range(current_layer_position+1):
                node = self.Tree[layer][id_node]
                node.hd_message = torch.zeros((self.variable_type, self.batch_size), device=self.device)
                for id_child in range(self.n_child):
                    if id_node * self.n_child + id_child <= previous_layer_position:
                        node.hd_message += node.children[id_child].qd_message
                node.qd_message = torch.log(torch.mm(self.transition[layer-1][id_node], torch.exp(node.hd_message)))

        # Update the root node 
        self.root_node.hd_message = sum(self.Tree[1][id_child].qd_message 
                                        for id_child in range(current_layer_position+1))
        self.root_node.bu_message = self.root_node.hd_message 

        if external_hd_message is not None:
            self.root_node.bu_message += external_hd_message

        # Upward process
        for layer in range(1, self.n_layer+1):
            node = self.Tree[layer][object_parents[-layer]]
            if share_parents_flags[-layer]:
                diff = node.parent.bu_message - node.qd_message
                node.bu_message = node.hd_message + torch.log(torch.mm(self.transition[layer-1][object_parents[-layer]].T, torch.exp(diff)))
            else:
                node.bu_message = torch.log(torch.mm(self.transition[layer-1][object_parents[-layer]].T, torch.exp(node.parent.bu_message)))
            node.bu_message -= torch.max(node.bu_message, dim=0)[0]

        self.NWP_flag = True
        return torch.exp(node.bu_message)/torch.sum(torch.exp(node.bu_message), 0)

    def BP_NWP_autoregressive(self, guide_info=False, external_hd_message=None, verbose=False, pos=3):
        """
        Belief propagation for the next word prediction task. 

        Args:
        guide_info: bool, whether to generate guided information
        device: str, device to run the model
        external_hd_message: tensor, external information for the hd message
        verbose: bool, whether to print verbose information
        pos: int, position to print verbose information

        Returns:
        predictions: torch.tensor, predictions for the next word
        predict_pp: torch.tensor, posterior probability for the next word
        guided_layers: list of torch.tensor, guided information for each layer
        """
        num_leaves = len(self.Tree[-1])
        guided_layers = []

        if guide_info:
            guided_layers.append(torch.zeros((self.batch_size, num_leaves-1, self.variable_type), device=self.device))
            for iter in range(self.n_layer):
                guided_layers.append(torch.zeros((self.batch_size, num_leaves-1, 2*self.variable_type), device=self.device))
            for iter in range(self.n_layer):
                guided_layers.append(torch.zeros((self.batch_size, num_leaves-1, self.variable_type), device=self.device))

        predict_pp = torch.zeros((self.batch_size, num_leaves-1, self.variable_type), device=self.device)

        # Generate the message iteratively
        for position in range(num_leaves-1):
            id_node = position
            node = self.Tree[-1][position]
            goal_node = self.Tree[-1][position+1]
            node.qd_message = torch.log(self.transition[-1][id_node][:, self.T_value[-1][id_node]])
            node.qd_message -= node.qd_message.max(0)[0]

            if verbose and position == pos:
                print("Leave", node.qd_message.T)
                print("Downard Process")

            if guide_info:
                guided_layers[0][:,position,:] = node.qd_message.T
                
            goal_node_parents = [goal_node]
            share_parents_flags = [False]
            id_goal_nodes = [position+1]

            # Downward process
            for layer in range(self.n_layer-1, 0, -1):
                parent_id_node = id_node // self.n_child
                parent_node = node.parent
                parent_node.hd_message = torch.zeros((self.variable_type, self.batch_size), device=self.device)
                
                for id_child in range(self.n_child):
                    if id_child + parent_id_node*self.n_child <= id_node:
                        parent_node.hd_message += parent_node.children[id_child].qd_message
                        
                parent_node.hd_message -= parent_node.hd_message.max(0)[0]
                parent_node.qd_message = torch.log(torch.mm(self.transition[layer-1][parent_id_node], torch.exp(parent_node.hd_message)))
                parent_node.qd_message -= parent_node.qd_message.max(0)[0]

                if verbose and position == pos:
                    print("Layer", layer, parent_node.qd_message.T)

                if guide_info:
                    guided_layers[self.n_layer-layer][:,position,:self.variable_type] = parent_node.hd_message.T
                    guided_layers[self.n_layer-layer][:,position,self.variable_type:] = parent_node.qd_message.T

                goal_node_parents.append(goal_node_parents[-1].parent)
                id_goal_nodes.append(id_goal_nodes[-1]//self.n_child)
                node = parent_node
                id_node = parent_id_node
                
                if node == goal_node_parents[-1]:
                    share_parents_flags.append(True)
                else:
                    share_parents_flags.append(False)

            # Update the root node
            self.root_node.hd_message = torch.zeros((self.variable_type, self.batch_size), device=self.device)
            for id_child in range(self.n_child):
                if id_child <= id_node:
                    self.root_node.hd_message += self.root_node.children[id_child].qd_message
                    
            self.root_node.hd_message -= self.root_node.hd_message.max(0)[0]
            self.root_node.bu_message = self.root_node.hd_message

            if external_hd_message is not None:
                self.root_node.bu_message += external_hd_message
                
            if verbose and position == pos:
                print("Root", self.root_node.hd_message.T)
                print("Root", self.root_node.bu_message.T)
                print("Upward Process")

            self.root_node.bu_message -= self.root_node.bu_message.max(0)[0]

            if guide_info:
                guided_layers[self.n_layer][:,position,:self.variable_type] = self.root_node.hd_message.T
                guided_layers[self.n_layer][:,position,self.variable_type:] = self.root_node.bu_message.T

            # Upward process
            for layer in range(1, self.n_layer+1):
                node = goal_node_parents[-layer]
                if share_parents_flags[-layer]:
                    diff = node.parent.bu_message - node.qd_message
                    node.bu_message = node.hd_message + torch.log(torch.mm(self.transition[layer-1][id_goal_nodes[-layer]].T, torch.exp(diff)))
                else:
                    node.bu_message = torch.log(torch.mm(self.transition[layer-1][id_goal_nodes[-layer]].T, torch.exp(node.parent.bu_message)))
                node.bu_message -= node.bu_message.max(0)[0]
                
                if verbose and position == pos:
                    print("Layer", self.n_layer - layer, node.bu_message.T)

                if guide_info:
                    guided_layers[self.n_layer+layer][:,position,:] = node.bu_message.T

            # print("node.bu_message shape:", node.bu_message.shape)
            # print("predict_pp shape:", predict_pp.shape)
            # print("position value:", position)

            predict_pp[:,position,:] = (torch.exp(node.bu_message)/torch.sum(torch.exp(node.bu_message), 0)).T


        return predict_pp, guided_layers

    def BP_DNS(self, z, sigma=1.0, external_hd_message=None):
        """
        Belief propagation for denoising tasks using PyTorch.
        """
        self.posterior_mean_DNS = torch.zeros_like(z.T, device=self.device)
        
        # Convert inputs to tensors if they aren't already
        z = torch.tensor(z, device=self.device) if not isinstance(z, torch.Tensor) else z
        vt_choices = torch.linspace(0, self.variable_type - 1, self.variable_type, device=self.device)

        # Generate message for leaf nodes
        for id_node, node in enumerate(self.Tree[-1]):
            node.hd_message = -0.5 * (z.T[id_node].unsqueeze(0) - vt_choices.unsqueeze(1)) ** 2 / (sigma ** 2)
            # print("node.hd_message shape:", node.hd_message.shape)
            node.qd_message = torch.log(torch.mm(self.transition[-1][id_node], torch.exp(node.hd_message)))

        # Downward process
        for layer in range(self.n_layer-1, 0, -1):
            for id_node, node in enumerate(self.Tree[layer]):
                node.hd_message = sum(child.qd_message for child in node.children)
                node.hd_message -= torch.max(node.hd_message, dim=0)[0]
                node.qd_message = torch.log(torch.mm(self.transition[layer-1][id_node], torch.exp(node.hd_message)))

        # Update root node
        self.root_node.hd_message = sum(child.qd_message for child in self.root_node.children)
        self.root_node.hd_message -= torch.max(self.root_node.hd_message, dim=0)[0]
        self.root_node.bu_message = self.root_node.hd_message

        if external_hd_message is not None:
            self.root_node.bu_message += torch.tensor(external_hd_message, device=self.device)

        # Upward process
        for layer in range(1, self.n_layer+1):
            for id_node, node in enumerate(self.Tree[layer]):
                diff = node.parent.bu_message - node.qd_message
                trans_prob = self.transition[layer-1][id_node].T
                node.bu_message = node.hd_message + torch.log(torch.mm(trans_prob, torch.exp(diff)))
                node.bu_message -= torch.max(node.bu_message, dim=0)[0]

        # Calculate posterior mean
        for idx, node in enumerate(self.leaves_nodes):
            probs = torch.exp(node.bu_message)
            probs = probs / probs.sum(dim=0)
            # print("vt_choices shape:", vt_choices.shape)
            # print("probs shape:", probs.shape)
            self.posterior_mean_DNS[idx] = torch.mm(vt_choices.unsqueeze(0), probs).squeeze()
        
        self.posterior_mean_DNS = self.posterior_mean_DNS.T

        self.dns_flag = True
        return self.posterior_mean_DNS        

    
    def guided_info(self, device=None):
        """
        Guided information for the tree using PyTorch tensors.
        
        Args:
            device: str or torch.device, device to place tensors on. If None, uses self.device
        """
        if device is None:
            device = self.device
        
        guided_layers = []
        extend_time = 1
        
        if self.cls_flag:            
            for i in range(self.n_layer-1, -1, -1):
                extend_time *= self.n_child
                guided_layer = []

                for node in self.Tree[i]:
                    if node.hd_message is not None:
                        # Clone the tensor instead of copying
                        guided_layer.extend([node.hd_message.clone() for _ in range(extend_time)])

                # Stack tensors directly without converting to numpy
                guided_layer = torch.stack(guided_layer)
                guided_layer = guided_layer.permute(2, 0, 1)  # Equivalent to double transpose
                # guided_layer = guided_layer.to(device)
                guided_layers.append(guided_layer)
        
        elif self.dns_flag:
            for i in range(self.n_layer, 0, -1):
                guided_layer_h = []
                guided_layer_q = []
                
                for node in self.Tree[i]:
                    if node.hd_message is not None:
                        guided_layer_h.extend([node.hd_message.clone() for _ in range(extend_time)])
                        guided_layer_q.extend([node.qd_message.clone() for _ in range(extend_time)])
                    
                # Concatenate tensors directly
                guided_layer = torch.cat([
                    torch.stack(guided_layer_h),
                    torch.stack(guided_layer_q)
                ], dim=1)
                guided_layer = guided_layer.permute(2, 0, 1)
                # guided_layer = guided_layer.to(device)
                guided_layers.append(guided_layer)
                extend_time *= self.n_child

            # Handle root node
            guided_layer_h = [self.root_node.hd_message.clone() for _ in range(extend_time)]
            guided_layer_q = [self.root_node.bu_message.clone() for _ in range(extend_time)]
            guided_layer = torch.cat([
                torch.stack(guided_layer_h),
                torch.stack(guided_layer_q)
            ], dim=1)
            guided_layer = guided_layer.permute(2, 0, 1)
            # guided_layer = guided_layer.to(device)
            guided_layers.append(guided_layer)

            for i in range(self.n_layer):
                extend_time = extend_time // self.n_child
                guided_layer_h = []
                guided_layer_q = []
                guided_layer_b = []
                
                for node in self.Tree[i+1]:
                    if node.hd_message is not None:
                        guided_layer_h.extend([node.hd_message.clone() for _ in range(extend_time)])
                        guided_layer_q.extend([node.qd_message.clone() for _ in range(extend_time)])
                        guided_layer_b.extend([node.bu_message.clone() for _ in range(extend_time)])
                    
                # Concatenate tensors directly
                guided_layer = torch.cat([
                    torch.stack(guided_layer_h),
                    torch.stack(guided_layer_q),
                    torch.stack(guided_layer_b)
                ], dim=1)
                guided_layer = guided_layer.permute(2, 0, 1)
                # guided_layer = guided_layer.to(device)
                guided_layers.append(guided_layer)
        
        return guided_layers
            

    @property
    def leaves_nodes(self):
        return self.Tree[-1]
    
    @property
    def root_node(self):
        return self.Tree[0][0]

    @property
    def leaves_values(self):
        return torch.stack(self.T_value[-1])

    @property
    def root_value(self):
        return self.T_value[0][0]

# Sampler Classes

class SingleSampler:
    """
    Single sampler for the GHM tree. 
    """
    def __init__(self, 
                 n_layer=3, 
                 n_child=4, 
                 p_y=torch.ones(10)/10, 
                 p_flip=0.3, 
                 flip_scale=1.0, 
                 variable_type=10, 
                 translation_invariance=True, 
                 seedtree=42, 
                 sigma=1,
                 device='cpu'):
        self.n_layer = n_layer # number of layers
        self.n_child = n_child # number of children
        self.p_y = p_y # prior probability of each class
        self.p_flip = p_flip # flip probability
        self.translation_invariance = translation_invariance # translation invariance
        self.seedtree = seedtree # seed for the tree
        self.flip_scale = flip_scale # flip scale
        self.device = device # device
        self.variable_type = variable_type # number of classes
        self.sigma = sigma # sigma

        # Generate the transition kernel
        self.transition = GenTransition(n_layer=n_layer, 
                                        n_child=n_child, 
                                        variable_type=variable_type, 
                                        p_flip=p_flip, 
                                        flip_scale=flip_scale,
                                        translation_invariance=translation_invariance, 
                                        device=device, 
                                        seedtree=seedtree)
    
    def get_batch(self, batch_size=128, job='clas', guide=False):

        # Generate the tree
        tree = GHMTree(n_layer=self.n_layer, 
                    n_child=self.n_child, 
                    variable_type=self.variable_type, 
                    p_y=self.p_y, 
                    p_flip=self.p_flip, 
                    transition=self.transition, 
                    batch_size=batch_size, 
                    build_tree=True, 
                    device=self.device)     
        
        # Get the leaves values
        leaves_values = tree.leaves_values.T
        root_values = tree.root_value
        guided_info = []
        pp = None

        # Get the info for different jobs 
        if job == 'clas': # classification task
            inputs = leaves_values
            targets = root_values

            if guide: # Collect the guided info and posterior probability
                tree.BP_CLS()
                guided_info = tree.guided_info(device=self.device)
                pp = tree.posterior_probability_CLS
        
        elif job == 'dns': # denoising task
            inputs = leaves_values + torch.randn_like(leaves_values.float()) * self.sigma 
            targets = leaves_values

            if guide: # Collect the guided info and posterior probability
                tree.BP_DNS(z=inputs, sigma=self.sigma)
                guided_info = tree.guided_info(device=self.device)
                pp = tree.posterior_mean_DNS
        
        elif job == 'nwp': # next word prediction task
            inputs = leaves_values[:,:-1]
            targets = leaves_values[:,1:]

            if guide: # Collect the guided info and posterior probability
                pp,guided_info = tree.BP_NWP_autoregressive(guide_info=True)

        return inputs, targets, guided_info, pp
    
    def get_Bayes(self, n_eval=10000, job='clas'):
        """
        Get the Bayesian Error.
        """
        res = self.get_batch(batch_size=n_eval, guide=True, job=job)

        if job == 'clas':
            predict_pp = res[-1]
            root = res[1]
            select_pp = predict_pp[range(n_eval), root]
            loss = -torch.log(select_pp)
        elif job == 'dns':
            pred = res[-1]
            target = res[1]
            loss = torch.sum(torch.pow(pred - target, 2), dim=1)
        elif job == 'nwp':
            pred = res[3] 
            target = res[1]
            pred = pred.reshape(-1, self.variable_type)
            target_c = target.reshape(-1).long()
            loss = -torch.log(pred[torch.arange(len(target_c), device=self.device), target_c])

        return loss.mean().item(), loss.std().item() / np.sqrt(n_eval)
        

class DoubleSampler:
    """
    Double sampler for the GHM tree. One is the image sampler and the other is the text sampler. 
    """
    def __init__(self, 
                 n_layers=(4,4), 
                 n_childs=(3,3), 
                 p_ys=(torch.ones(10)/10, torch.ones(10)/10), 
                 p_flips=(0.3, 0.3), 
                 flip_scale=1, 
                 variable_type = 10, 
                 translation_invariance=True, 
                 seedtrees = [42, 42], 
                 sigma=1,
                 device='cpu'):
        
        self.n_layers = n_layers # number of layers for each tree
        self.n_childs = n_childs # number of children for each tree
        self.p_ys= p_ys # prior probability of each class for each tree
        self.flip_scale = flip_scale # flip scale for each tree
        self.variable_type = variable_type # number of classes for each tree
        self.seedtrees = seedtrees # seed for each tree
        self.device = device # device for each tree
        self.p_flips = p_flips # flip probability for each tree
        self.sigma = sigma # sigma for each tree

        # Generate the transition kernels
        self.t_transition = GenTransition(n_layer=n_layers[0], 
                                          n_child=n_childs[0], 
                                          variable_type=variable_type, 
                                          p_flip=p_flips[0], 
                                          flip_scale=flip_scale, 
                                          translation_invariance=translation_invariance, 
                                          device=device,
                                          seedtree=seedtrees[0])

        self.i_transition = GenTransition(n_layer=n_layers[1], 
                                          n_child=n_childs[1], 
                                          variable_type=variable_type, 
                                          p_flip=p_flips[1], 
                                          flip_scale=flip_scale, 
                                          translation_invariance=translation_invariance, 
                                          device=device, seedtree=seedtrees[1])
    
    def get_batch(self, batch_size=128,job='clip', guide=False):
        
        root_tree = torch.randint(0, self.variable_type, (batch_size,), device=self.device)

        # Generate the text tree
        text_tree = GHMTree(n_layer=self.n_layers[0], 
                            n_child=self.n_childs[0], 
                            variable_type=self.variable_type, 
                            p_y=self.p_ys[0], 
                            p_flip=self.p_flips[0], 
                            transition=self.t_transition, 
                            batch_size=batch_size, 
                            build_tree=True, 
                            root=root_tree, 
                            device=self.device)

        # Generate the image tree
        image_tree = GHMTree(n_layer=self.n_layers[1], 
                             n_child=self.n_childs[1], 
                             variable_type=self.variable_type, 
                             p_y=self.p_ys[1], 
                             p_flip=self.p_flips[1], 
                             transition=self.i_transition, 
                             batch_size=batch_size, 
                             build_tree=True, 
                             root=root_tree, 
                             device=self.device)
        
        # Get the leaves values
        text_tree_leaves = text_tree.leaves_values.T
        image_tree_leaves = image_tree.leaves_values.T
        tree_root = text_tree.root_value
        text_guided_info = []
        image_guided_info = []
        text_pp = None
        image_pp = None

        # Collect the guided info and posterior probability
        if job=='clip': # clip model vs clip model

            input_text = text_tree_leaves
            input_image = image_tree_leaves
            target_text = tree_root
            target_image = tree_root

            if guide:
            
                text_tree.BP_CLS()
                image_tree.BP_CLS()

                text_guided_info = text_tree.guided_info(device=self.device)
                image_guided_info = image_tree.guided_info(device=self.device)
                text_pp = text_tree.posterior_probability_CLS
                image_pp = image_tree.posterior_probability_CLS

        
        elif job=="nwp-dns": # dns model vs nwp model

            input_text = text_tree_leaves[:,:-1]
            target_text = text_tree_leaves[:,1:]
            input_image = image_tree_leaves + torch.randn_like(image_tree_leaves.float()) * self.sigma 
            target_image = image_tree_leaves 

            if guide:
            
                image_tree.BP_DNS(z=input_image, sigma=self.sigma)
                image_guided_info = image_tree.guided_info(device=self.device) 
                image_pp = image_tree.posterior_mean_DNS
                text_pp, text_guided_info = text_tree.BP_NWP_autoregressive(guide_info=True)

        return [input_text, target_text, text_guided_info, text_pp], [input_image, target_image, image_guided_info, image_pp]
    
    def get_bayes(self, n_eval=5000, job='clip', batch_size=1000, verbose=False):
        """
        Get the Bayesian Error.
        """
        if job == 'clip':
            text_batch, image_batch = self.get_batch(batch_size=n_eval, guide=True, job=job)
            text_pp, image_pp = text_batch[3], image_batch[3] 
            losses = torch.zeros(n_eval, device=self.device)
            assert n_eval % batch_size == 0, "n_eval must be divisible by batch_size"
            n_batch = n_eval // batch_size
            for i in range(n_batch):
                similarity = text_pp[i*batch_size:(i+1)*batch_size] @ (image_pp[i*batch_size:(i+1)*batch_size].T)  # (batch_size, batch_size) size matrix 
                loss_text_image = torch.diag(similarity)/torch.sum(similarity, dim=1) 
                loss_image_text = torch.diag(similarity)/torch.sum(similarity, dim=0) 
                if verbose:
                    print("loss_text_image: ", loss_text_image[:20])
                    print("log_loss_text_image: ", -torch.log(loss_text_image)[:20])
                    print("loss_image_text: ", loss_image_text[:20])
                    print("log_loss_image_text: ", -torch.log(loss_image_text)[:20])
                losses[i*batch_size:(i+1)*batch_size] = -torch.log(loss_text_image) - torch.log(loss_image_text) 

        return losses.mean().item(), losses.std().item() / np.sqrt(n_eval)   
    
# Single Sampler Class
class ClassificationSampler(SingleSampler):
    def __init__(self, n_layer, n_child, p_y, p_flip=0.3, flip_scale=1, variable_type=10,translation_invariance=True, seedtree=42, device='cpu'):

        super().__init__(n_layer, n_child, p_y, p_flip,flip_scale, variable_type, translation_invariance, seedtree, device) 

    def get_batch(self, batch_size=128, guide=False):
        tree = GHMTree(self.n_layer, self.n_child, self.variable_type, self.p_y, self.p_flip, self.transition, batch_size, build_tree=True, device=self.device)
        leaves_values = tree.leaves_values.T
        root_values = tree.root_value
        if guide:  
            tree.BP_CLS()
            guided_info = tree.guided_info()
        else:
            guided_info = None 
        
        return leaves_values, root_values, guided_info, tree.posterior_probability_CLS
    
    def get_Bayes(self,n_eval=10000):
        """
        Get the Bayesian Error.
        """
        res = self.get_batch(batch_size=n_eval, guide=True)
        predict_pp = res[-1]
        predict_pp = torch.tensor(predict_pp, dtype=torch.float)
        root = res[1]
        select_pp = predict_pp[range(n_eval), root]
        # print(select_pp[:10])
        loss = -torch.log(select_pp)


        return loss.mean().item(), loss.std().item() / np.sqrt(n_eval)
    
class DenoiseSampler(SingleSampler):

    def __init__(self, n_layer, n_child, p_y, p_flip=0.3, sigma=1, flip_scale=1, variable_type=10,translation_invariance=True, seedtree=42, device='cpu'):

        super().__init__(n_layer, n_child, p_y, p_flip,flip_scale, variable_type, translation_invariance, seedtree, device)
        self.sigma = sigma
    
    def get_batch(self, batch_size=128, guide=False):
        tree = GHMTree(self.n_layer, self.n_child, self.variable_type, self.p_y, self.p_flip, self.transition, batch_size, build_tree=True, device=self.device)
        zs = torch.randn(self.n_child**self.n_layer, batch_size, device=self.device) * self.sigma + tree.leaves_values
        xs = tree.leaves_values
        if guide:
            tree.BP_DNS(zs, self.sigma)
            guided_info = tree.guided_info()
        else:
            guided_info = None 
        
        return zs.T, xs.T, guided_info, tree.posterior_mean_DNS
    
    def get_bayes(self, n_eval=10000):
        res = self.get_batch(batch_size=n_eval, guide=True)
        pred = res[-1]
        target = res[1]
        loss = torch.sum(torch.pow(pred - target, 2), dim=1)
        return loss.mean().item(), loss.std().item() / np.sqrt(n_eval)

class NWPSampler(SingleSampler):
    def __init__(self, n_layer, n_child, p_y, p_flip=0.3, flip_scale=1, variable_type=10,translation_invariance=True, seedtree=42, device='cpu'):
        super().__init__(n_layer, n_child, p_y, p_flip,flip_scale, variable_type, translation_invariance, seedtree, device)

    def get_batch(self, batch_size=128, guide=False):
        tree = GHMTree(self.n_layer, self.n_child, self.variable_type, self.p_y, self.p_flip, self.transition, batch_size, build_tree=True, device=self.device)
        tokens = tree.leaves_values.T
        if guide:
            BP_output,guided_info = tree.BP_NWP_autoregressive(guide_info=True)
        else:
            guided_info = None
            BP_output = None
        
        return tokens[:,:-1], tokens[:,1:], guided_info, BP_output  

    def get_bayes(self,n_eval=30000):
        """
        Get the Baysian Error. 
        """
        res = self.get_batch(batch_size=n_eval, guide=True)
        pred = res[3] 
        target = res[1]
        pred = pred.reshape(-1, self.variable_type)
        target_c = target.reshape(-1).long()
        loss = -torch.log(pred[torch.arange(len(target_c), device=self.device), target_c])
        return loss.mean().item(), loss.std().item() / np.sqrt(n_eval)   

# Double Sampler Class
class ClipSampler(DoubleSampler):

    def __init__(self, n_layers, n_childs, p_ys, p_flips, K=4, flip_scale=1, variable_type=10, translation_invariance=True, seedtrees=[42, 42], device='cpu'):
        super().__init__(n_layers, n_childs, p_ys, p_flips, flip_scale, variable_type, translation_invariance, seedtrees, device=device)
        self.K = K
        self.device = device
    
    def get_batch(self, batch_size=128, guide=False):
            text_tree_root = torch.randint(0, self.variable_type, (batch_size * (self.K+1),), device=self.device)
            image_tree_root = torch.randint(0, self.variable_type, (batch_size * (self.K-1),), device=self.device)
            image_tree_root = torch.cat([text_tree_root[:2*batch_size], image_tree_root], dim=0)

            text_tree = GHMTree(self.n_layers[0], self.n_childs[0], self.variable_type, self.p_ys[0], self.p_flips[0], self.t_transition, batch_size*(self.K+1), build_tree=True, root=text_tree_root, device=self.device)

            image_tree = GHMTree(self.n_layers[1], self.n_childs[1], self.variable_type, self.p_ys[1], self.p_flips[1], self.i_transition, batch_size*(self.K+1), build_tree=True, root=image_tree_root, device=self.device)

            if guide:
                text_tree.BP_CLS()
                image_tree.BP_CLS()
                text_guided_info = text_tree.guided_info(device=self.device)
                image_guided_info = image_tree.guided_info(device=self.device)
                t_pp = text_tree.posterior_probability_CLS
                i_pp = image_tree.posterior_probability_CLS   
            else:
                text_guided_info = None
                image_guided_info = None
                t_pp = None
                i_pp = None
            
            image_tree_leaves = image_tree.leaves_values.T
            text_tree_leaves = text_tree.leaves_values.T
            image_tree_root = image_tree_root
            text_tree_root = text_tree_root

            return [text_tree_leaves, text_tree_root, text_guided_info, t_pp], [image_tree_leaves, image_tree_root, image_guided_info, i_pp]
    
    def get_Bayes(self,n_eval=10000):
        """
        Get the Bayesian Error.
        """
        batch_size = 1000 
        S_total = torch.zeros(n_eval, device=self.device)
        for i in range(n_eval // batch_size):
            res = self.get_batch(batch_size=batch_size, guide=True)
            ttree_pp = res[0][3].T
            itree_pp = res[1][3].T

            # K image and 1 text 
            t_pp_match = ttree_pp[:, :batch_size] 
            i_pp_match = itree_pp[:, :batch_size] 
            t_pp_indep = ttree_pp[:, 2*batch_size:] 

            # compute S 
            S_match = torch.sum(t_pp_match * i_pp_match, dim=0) * self.variable_type
            S_indep = torch.sum(t_pp_indep * i_pp_match.repeat(1, self.K-1), dim=0)
            concat_mat = torch.kron(torch.ones(self.K-1, 1, device=self.device), torch.eye(batch_size, device=self.device)).double()
            S_indep = torch.matmul(S_indep, concat_mat) * self.variable_type
            S = -torch.log(S_match / (S_indep + S_match))

            # K text and 1 image 
            t_pp_match = ttree_pp[:, batch_size:2*batch_size] 
            i_pp_match = itree_pp[:, batch_size:2*batch_size]
            i_pp_indep = itree_pp[:, 2*batch_size:] 

            # compute S 
            S_match = torch.sum(t_pp_match * i_pp_match, dim=0) * self.variable_type
            S_indep = torch.sum(i_pp_indep * t_pp_match.repeat(1, self.K-1), dim=0)
            S_indep = torch.matmul(S_indep, concat_mat) * self.variable_type
            S += -torch.log(S_match / (S_indep + S_match))

            S_total[i*batch_size:(i+1)*batch_size] = S
        return torch.mean(S_total).item(), torch.std(S_total).item() / (n_eval ** 0.5)

def clip_loss_compute(ttree_pp, itree_pp, n_eval, K, variable_type):
        
        t_pp_match = ttree_pp[:, :n_eval] 
        i_pp_match = itree_pp[:, :n_eval] 
        t_pp_indep = ttree_pp[:, 2*n_eval:] 

        # compute S 
        S_match = np.sum(t_pp_match * i_pp_match, 0) * variable_type
        S_indep = np.sum(t_pp_indep * np.tile(i_pp_match, (1, K-1)),0)
        concat_mat = np.kron(np.ones([K-1,1]), np.eye(n_eval))
        S_indep = S_indep.dot(concat_mat)* variable_type 
        S = -np.log(S_match/(S_indep + S_match)) 

        # K text and 1 image 
        t_pp_match = ttree_pp[:, n_eval: 2*n_eval] 
        i_pp_match = itree_pp[:, n_eval: 2*n_eval]
        i_pp_indep = itree_pp[:, 2*n_eval:] 

        # compute S 
        S_match = np.sum(t_pp_match * i_pp_match, 0) *variable_type
        S_indep = np.sum(i_pp_indep * np.tile(t_pp_match, (1, K-1)),0)
        S_indep = S_indep.dot(concat_mat) *variable_type 
        S += -np.log(S_match/(S_indep + S_match)) 

        return np.mean(S), np.std(S) / np.sqrt(n_eval)

class ConditionalDenoiseSampler(DoubleSampler):
    """
    Conditional denoise sampler for the GHM tree.
    """
    def __init__(self, n_layers, n_childs, p_ys, p_flips, sigma=1, flip_scale=1, variable_type=10, translation_invariance=True, seedtree=42):
        super().__init__(n_layers, n_childs, p_ys, p_flips, flip_scale, variable_type, translation_invariance, seedtree)
        self.sigma = sigma
    
    def get_batch(self, batch_size=128, device="cpu", guide=False):
        tree_root = np.random.choice(self.variable_type, size = batch_size)
        
        text_tree = GHMTree(self.n_layers[0], self.n_childs[0], self.variable_type, self.p_ys[0], self.p_flips[0], self.t_transition, batch_size, build_tree=True, root=tree_root)
        image_tree = GHMTree(self.n_layers[1], self.n_childs[1], self.variable_type, self.p_ys[1], self.p_flips[1], self.i_transition, batch_size, build_tree=True, root=tree_root)

        image_tree_leaves = torch.tensor(image_tree.leaves_values, dtype=torch.long).T.to(device)
        text_tree_leaves = torch.tensor(text_tree.leaves_values, dtype=torch.long).T.to(device)
        # image_tree_root = torch.tensor(image_tree.root_value, dtype=torch.long).to(device)
        text_tree_root = torch.tensor(text_tree.root_value, dtype=torch.long).to(device)
        image_tree_noise = np.random.randn(self.n_childs[1]**self.n_layers[1], batch_size) * self.sigma + image_tree.leaves_values 

        if guide:
            text_tree.BP_CLS()
            external_hd_message = text_tree.root_node.hd_message
            image_tree.BP_DNS(image_tree_noise, self.sigma, external_hd_message=external_hd_message)
            text_guided_info = text_tree.guided_info(device=device)
            image_guided_info = image_tree.guided_info(device=device)
        else:
            text_tree.BP_CLS()
            external_hd_message = text_tree.root_node.hd_message
            image_tree.BP_DNS(image_tree_noise, self.sigma, external_hd_message=external_hd_message)
            text_guided_info = None
            image_guided_info = None
        
        image_tree_noise = torch.tensor(image_tree_noise, dtype=torch.float32).T.to(device)

        return (text_tree_leaves, text_tree_root, text_guided_info, text_tree.posterior_probability_CLS), (image_tree_noise, image_tree_leaves, image_guided_info, image_tree.posterior_mean_DNS.T)
    
    def get_Bayes(self,n_eval=30000):
        """
        Get the Baysian Error. 
        """
        res = self.get_batch(batch_size=n_eval, guide=True)
        pred = res[1][3] 
        target = res[1][1].numpy()
        loss = np.sum(np.power(pred-target,2),1)
        return np.mean(loss), np.std(loss) / np.sqrt(n_eval)

class NextWordPredictSampler(DoubleSampler):
    def __init__(self, n_layers, n_childs, p_ys, p_flips, flip_scale=1, variable_type=10, translation_invariance=True, seedtree=42):
        super().__init__(n_layers, n_childs, p_ys, p_flips, flip_scale, variable_type, translation_invariance, seedtree)
    
    def get_batch(self, batch_size=128, device="cpu", guide=False):
        tree_root = np.random.choice(self.variable_type, size = batch_size)
        
        text_tree = GHMTree(self.n_layers[0], self.n_childs[0], self.variable_type, self.p_ys[0], self.p_flips[0], self.t_transition, batch_size, build_tree=True, root=tree_root)
        image_tree = GHMTree(self.n_layers[1], self.n_childs[1], self.variable_type, self.p_ys[1], self.p_flips[1], self.i_transition, batch_size, build_tree=True, root=tree_root)

        image_tree_leaves = torch.tensor(image_tree.leaves_values, dtype=torch.long).T.to(device)
        image_tree_roots = torch.tensor(image_tree.root_value, dtype=torch.long).to(device)
        text_tree_leaves = torch.tensor(text_tree.leaves_values, dtype=torch.long).T.to(device)
        input_text_tree_sequence = text_tree_leaves[:,:-1]
        target_text_tree_sequence = text_tree_leaves[:,1:]

        if guide:
            image_tree.BP_CLS()
            external_hd_message = image_tree.root_node.hd_message
            BP_output,text_guided_info = text_tree.BP_NWP_autoregressive(external_hd_message=external_hd_message, device=device, guide_info=True)
            image_guided_info = image_tree.guided_info(device=device)
        else:
            image_tree.BP_CLS()
            external_hd_message = image_tree.root_node.hd_message
            BP_output,_ = text_tree.BP_NWP_autoregressive(external_hd_message=external_hd_message, device=device, guide_info=False)
            image_guided_info = None
            text_guided_info = None
        
        return (input_text_tree_sequence, target_text_tree_sequence, text_guided_info, BP_output), (image_tree_leaves, image_tree_roots, image_guided_info, image_tree.posterior_probability_CLS)

    def get_Bayes(self,n_eval=30000):
        """
        Get the Baysian Error. 
        """
        res = self.get_batch(batch_size=n_eval, guide=True)
        pred = res[0][-1] 
        target = res[0][1]
        pred = pred.reshape(-1, self.variable_type)
        target_c = target.reshape(-1)
        loss = -np.log(pred[range(len(target_c)), target_c])
        # print(loss)
        return np.mean(loss), np.std(loss) / np.sqrt(n_eval)


class Augmentator(nn.Module):
    def __init__(self, crop_ratio_range=(0.6, 0.9), sigma_range=(0.1, 2.0)):
        super().__init__()
        self.crop_ratio_range = crop_ratio_range
        self.sigma_range = sigma_range
    
    def forward(self,x):
        x = random_crop_1d(x, self.crop_ratio_range) 
        # x = gaussian_blur_1d(x, self.sigma_range) 
        return x

# Augmentation 1 random crop
def random_crop_1d(x, crop_ratio_range=(0.6, 0.9)):
    """Randomly crop 1D sequences with independent crop ratios and resize back to original length.
    Also has 50% chance to flip each sequence."""
    batch_size, seq_len = x.shape
    
    # Generate random crop ratios for each sequence
    crop_ratios = torch.empty(batch_size).uniform_(crop_ratio_range[0], crop_ratio_range[1])
    crop_lens = (seq_len * crop_ratios).long()
    
    # Generate random start indices for each sequence
    start_idx = [torch.randint(0, seq_len - crop_len, (1,)).item() for crop_len in crop_lens]
    
    # Crop each sequence independently and resize
    resized = torch.zeros_like(x, dtype=x.dtype)
    for i in range(batch_size):
        # Crop
        cropped = x[i, start_idx[i]:start_idx[i]+crop_lens[i]]
        
        # Resize back to original length using linear interpolation
        x_idx = torch.linspace(0, crop_lens[i]-1, crop_lens[i])
        new_idx = torch.linspace(0, crop_lens[i]-1, seq_len)
        resized[i] = torch.from_numpy(
            np.interp(new_idx.numpy(), x_idx.numpy(), cropped.cpu().numpy())
        ).to(x.device)
        
        # 50% chance to flip
        if torch.rand(1).item() < 0.5:
            resized[i] = torch.flip(resized[i], dims=[0])
    
    return resized

# Agumentation 2 gaussian blur 
def gaussian_blur_1d(x, sigma_range=(0.1, 2.0)):
    """Apply Gaussian blur to 1D sequences with random sigma for each sequence, 50% chance to blur each sequence"""
    batch_size, seq_len = x.shape
    
    # Generate random sigma values for each sequence
    sigmas = np.random.uniform(sigma_range[0], sigma_range[1], size=batch_size)
    
    # Create kernel sizes based on sigma (odd numbers)
    kernel_sizes = (2 * np.ceil(3 * sigmas) + 1).astype(int)
    
    # Randomly decide which sequences to blur (50% chance)
    blur_mask = np.random.random(batch_size) < 0.5
    
    # Apply Gaussian blur to selected sequences
    blurred = np.zeros_like(x)
    for i in range(batch_size):
        if blur_mask[i]:
            # Create Gaussian kernel
            kernel_size = kernel_sizes[i]
            kernel = np.exp(-np.arange(-(kernel_size//2), kernel_size//2 + 1)**2 / (2*sigmas[i]**2))
            kernel = kernel / kernel.sum()  # Normalize
            
            # Apply convolution
            blurred[i] = np.convolve(x[i], kernel, mode='same')
        else:
            # Keep original sequence
            blurred[i] = x[i]
    
    return torch.from_numpy(blurred)

if __name__ == "__main__":
    device = "cuda:0"
    p_ys = torch.ones(10)/10 
    p_ys = p_ys.to(device).double()
    p_ys = [p_ys, p_ys]
    p_flips = [0.4, 0.4]
    
    sampler = DoubleSampler(n_layers=[4,4], n_childs=[3,3], p_ys=p_ys, p_flips=p_flips, flip_scale=1, variable_type=10, translation_invariance=True, seedtrees=[24, 42], device=device)
    bayes_error, bayes_error_std = sampler.get_bayes(n_eval=12800, job='clip', batch_size=256, verbose=False)
    print(bayes_error, bayes_error_std)

    sampler2 = ClipSampler(n_layers=[4,4], n_childs=[3,3], p_ys=p_ys, p_flips=p_flips, flip_scale=1, K=256, variable_type=10, translation_invariance=True, seedtrees=[24, 42], device=device)
    bayes_error, bayes_error_std = sampler2.get_Bayes(n_eval=128000) 
    print(bayes_error, bayes_error_std)

