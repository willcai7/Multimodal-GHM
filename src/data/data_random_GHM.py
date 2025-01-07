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
                  verbose=False):
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
    transition = [[] for _ in range(n_layer)]
    if verbose:
        skeleton = []
    if translation_invariance: 
        for layer in range(n_layer):
            skeleton_matrix =  np.identity(variable_type)[np.random.permutation(variable_type), :]
            invariance_transition = [(1 - p_flip) * skeleton_matrix + \
                    p_flip * _softmax_row(np.random.normal(0, flip_scale, [variable_type, variable_type])) for _ in range(n_child)]
            for _ in range(n_child ** layer):            
                transition[layer].extend(invariance_transition)
            if verbose:
                skeleton.append(skeleton_matrix)
    else:
        for layer in range(n_layer):
            for _ in range(n_child ** layer):
                transition[layer].extend([(1 - p_flip) * np.identity(variable_type)[np.random.permutation(variable_type), :] + \
                    p_flip * _softmax_row(np.random.normal(0, flip_scale, [variable_type, variable_type])) for _ in range(n_child)])
    if verbose:
        return transition, skeleton
    else:
        return transition

def _softmax_row(x = np.array([[]])):
    """
    Softmax function ignoring diagonal elements for each row of a matrix.
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

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
    GHM tree class.
    """
    def __init__(self, 
                 n_layer=4, 
                 n_child=3, 
                 variable_type = 10, 
                 p_y=np.ones(10)/10, 
                 p_flip=0.3, 
                 transition=None, 
                 batch_size=128, 
                 build_tree = False, 
                 root=None):
        
        self.variable_type = variable_type
        self.posterior_probability_CLS = None
        self.n_layer = n_layer
        self.n_child = n_child
        self.p_y = p_y
        self.p_flip = p_flip
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
        Generate values for the tree.
        """
        self.T_value = [[] for _ in range(self.n_layer + 1)]
        if self.root is not None: 
            self.T_value[0].append(self.root)
        else:
            self.T_value[0].append(np.random.choice(self.variable_type, 
                                                    size=self.batch_size, 
                                                    p=self.p_y))
        for layer in range(1, self.n_layer + 1):
            for id in range(len(self.T_value[layer - 1])):
                node_value = self.T_value[layer - 1][id]
                self.T_value[layer].extend([(np.random.rand(self.batch_size, 1) \
                        < self.transition[layer-1][id * self.n_child + id_child][node_value].cumsum(axis=1)).argmax(axis=1).tolist() for id_child in range(self.n_child)])
    
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
        Belief propagation for classification.
        """
        # generate message for the L-1 layer nodes 
        for id_node in range(len(self.Tree[-2])):
            node = self.Tree[-2][id_node]
            node.hd_message = np.zeros([self.variable_type, self.batch_size])
            # pdb.set_trace()
            for id_child in range(self.n_child):
                node.hd_message += np.log(self.transition[-1][id_node * self.n_child + id_child][:, self.T_value[-1][id_node*self.n_child+id_child]])
            node.hd_message -= np.max(node.hd_message,0)

        # generate message for layer L-2 to 0
        for layer in range(self.n_layer-2, -1, -1):
            for id_node in range(len(self.Tree[layer])):
                node = self.Tree[layer][id_node]
                node.hd_message = np.zeros([self.variable_type, self.batch_size])
                for id_child in range(self.n_child):
                    child = node.children[id_child]
                    node.hd_message += np.log(self.transition[layer][id_node * self.n_child + id_child] @ np.exp(child.hd_message))
                node.hd_message -= np.max(node.hd_message, 0)
        # final probability 

        # self.root_node.hd_message = np.zeros([self.variable_type, self.batch_size])
        h0 = self.root_node.hd_message + np.log(self.p_y).reshape(-1,1)
        h0 -= np.max(h0, 0)
        # print(h0)
        # pdb.set_trace()
        self.posterior_probability_CLS = np.exp(h0) / np.sum(np.exp(h0),0)

        self.cls_flag = True

        return self.posterior_probability_CLS    
    
    def BP_dummy_NWP(self, position, external_hd_message=None):
        """
        Belief propagation for denoising tasks. 
        """
        
        # print("Doing BP_dummy_NWP")
        # generate message for the leaf nodes 
        vt_choices = np.linspace(0, self.variable_type - 1, self.variable_type) # possible values for the variable
        index = 0
        for id_node in range(len(self.Tree[-1])):
            node = self.Tree[-1][id_node]
            if id_node >= position:
                node.hd_message = np.zeros([self.variable_type, 1])
                node.qd_message = np.log(self.transition[-1][id_node] @ np.exp(node.hd_message))
            else:
                node.hd_message = None
                node.qd_message = np.log(self.transition[-1][id_node][:, self.T_value[-1][id_node]])
            # print("Leave:", id_node, node.qd_message.T-np.min(node.qd_message).T)

        # downward process 
        for layer in range(self.n_layer-1, 0, -1):
            for id_node in range(len(self.Tree[layer])):
                node = self.Tree[layer][id_node]
                node.hd_message = sum( child.qd_message 
                                                for child in node.children)
                node.qd_message = np.log(self.transition[layer-1][id_node] @ np.exp(node.hd_message)) 
                # if id_node ==0:
                #     print("Layer:",layer, id_node, node.qd_message.T-np.min(node.qd_message).T)
        
        # Update the root node 
        self.root_node.hd_message = sum( child.qd_message 
                                                for child in self.root_node.children)
        self.root_node.bu_message = self.root_node.hd_message 
        # print("root",self.root_node.bu_message.T-np.min(self.root_node.bu_message).T)
        if external_hd_message is not None:
            self.root_node.bu_message += external_hd_message

        # upward process
        for layer in range(1, self.n_layer+1):
            for id_node in range(len(self.Tree[layer])):
                node = self.Tree[layer][id_node]
                diff = node.parent.bu_message - node.qd_message
                if node.hd_message is not None:
                    node.bu_message = node.hd_message + np.log(self.transition[layer-1][id_node].T @ np.exp(diff))
                    # if id_node ==0:
                    #     print("Layer:",layer, id_node, node.bu_message.T-np.min(node.bu_message).T)


        return np.exp(self.Tree[-1][position].bu_message)/np.sum(np.exp(self.Tree[-1][position].bu_message),0)        

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
            node.qd_message = np.log(self.transition[-1][id_node][:, self.T_value[-1][id_node]])
            print("Leave:", id_node, node.qd_message.T-np.max(node.qd_message).T)


        # downward process 
        for layer in range(self.n_layer-1, 0, -1):
            # print(layer)
            previous_layer_position = current_layer_position
            current_layer_position = (current_layer_position) // self.n_child 
            current_object_layer_position = (current_object_layer_position) // self.n_child
            share_parents_flags.append(current_layer_position == current_object_layer_position)
            object_parents.append(current_object_layer_position)
            for id_node in range(current_layer_position+1):
                # print(layer, id_node)
                node = self.Tree[layer][id_node]
                node.hd_message = np.zeros([self.variable_type, self.batch_size])
                for id_child in range(self.n_child):
                    # print(id_node * self.n_child + id_child, previous_layer_position)
                    if id_node * self.n_child + id_child <= previous_layer_position:
                        node.hd_message += node.children[id_child].qd_message
                node.qd_message = np.log(self.transition[layer-1][id_node] @ np.exp(node.hd_message)) 
                print("Layer:",layer, id_node, node.qd_message.T-np.max(node.qd_message).T)
        
        # Update the root node 
        # print(object_parents)
        # print(current_layer_position)
        # print(self.Tree[1][0].qd_message)
        # print(self.Tree[1][1].qd_message)
        self.root_node.hd_message = sum( self.Tree[1][id_child].qd_message 
                                                for id_child in range(current_layer_position+1))
        self.root_node.bu_message = self.root_node.hd_message 
        print("root",self.root_node.bu_message.T - np.max(self.root_node.bu_message).T)
        if external_hd_message is not None:
            self.root_node.bu_message += external_hd_message
        # upward process
        # print(object_parents)
        # print(share_parents_flags)
     
        for layer in range(1, self.n_layer+1):
            # print(layer)
            # print(object_parents[-layer])
            node = self.Tree[layer][object_parents[-layer]]
            if share_parents_flags[-layer]:
                diff = node.parent.bu_message - node.qd_message
                node.bu_message = node.hd_message + np.log(self.transition[layer-1][object_parents[-layer]].T @ np.exp(diff))
            else:
                node.bu_message = np.log(self.transition[layer-1][object_parents[-layer]].T @ np.exp(node.parent.bu_message))
            print("Layer:", self.n_layer - layer, node.bu_message.T-np.max(node.bu_message).T)
        self.NWP_flag = True
        return np.exp(node.bu_message)/np.sum(np.exp(node.bu_message),0)

    def BP_NWP_autoregressive(self, guide_info=False, device="cpu", external_hd_message=None, verbose=False, pos=3):
        """
        Belief propagation for the next word prediction task. 

        Args:
        guide_info: bool, whether to generate guided information
        device: str, device to run the model
        external_hd_message: numpy array, external information for the hd message

        Returns:
        predictions: torch.tensor, predictions for the next word
        predict_pp: torch.tensor, posterior probability for the next word
        guided_layers: list of torch.tensor, guided information for each layer
        """
        # generate message for the leaf nodes 
        vt_choices = np.linspace(0, self.variable_type - 1, self.variable_type) # possible values for the variable
        num_leaves = len(self.Tree[-1]) # number of leaves
        guided_layers = [] # guided information for each layer

        if guide_info: # Initialize the guided information 
            guided_layers.append(torch.tensor(np.zeros([self.batch_size,num_leaves-1, self.variable_type]), dtype=torch.float).to(device)) # guided information for the leaf nodes
            for iter in range(self.n_layer): # for layer 1 to n_layer+1, double output for hd and qd message
                guided_layers.append(torch.tensor(np.zeros([self.batch_size,(num_leaves-1), 2*self.variable_type]), dtype=torch.float).to(device))
            for iter in range(self.n_layer): # for layer n_layer+2 to 2*n_layer+1, single output for bu message
                guided_layers.append(torch.tensor(np.zeros([self.batch_size,(num_leaves-1), self.variable_type]), dtype=torch.float).to(device)) 

        predict_pp = torch.zeros([self.batch_size, num_leaves-1, self.variable_type]).to(device) # posterior probability for the next word

        # Generate the message iteratively
        for position in range(num_leaves-1):
            # print("Position:", position)
            id_node = position # id of the current position
            node = self.Tree[-1][position] # current node
            goal_node = self.Tree[-1][position+1] # goal node
            node.qd_message = np.log(self.transition[-1][id_node][:, self.T_value[-1][id_node]]) #initial qd message
            node.qd_message -= node.qd_message.max(0) # normalize the qd message

            if verbose and position == pos:
                print("Leave", node.qd_message.T)
                print("Downard Process")

            if guide_info:
                guided_layers[0][:,position,:] = torch.tensor(node.qd_message.T).to(device) # save the guided information
            goal_node_parents = [goal_node] # save the goal node ancester for future use
            share_parents_flags = [False] # whether the current node and the goal node share the same parent
            id_goal_nodes = [position+1] # id of the goal nodes to extract transition matrix
            
            # downward process
            for layer in range(self.n_layer-1, 0, -1): 
                parent_id_node = id_node// self.n_child # id of the parent node
                parent_node = node.parent # parent node
                # parent_node.hd_message += node.qd_message # update the hd message
                parent_node.hd_message = 0
                for id_child in range(self.n_child): # update the hd message
                    if id_child + parent_id_node*self.n_child <= id_node:
                        parent_node.hd_message += parent_node.children[id_child].qd_message
                parent_node.hd_message -= parent_node.hd_message.max(0) # normalize the hd message
                parent_node.qd_message = np.log(self.transition[layer-1][parent_id_node] @ np.exp(parent_node.hd_message)) # update the qd message
                parent_node.qd_message -= parent_node.qd_message.max(0) # normalize the qd message

                if verbose and position == pos:
                    print("Layer", layer, parent_node.qd_message.T)

                if guide_info: # save the guided information for the current layer
                    guided_layers[self.n_layer-layer][:,position,:self.variable_type] = torch.tensor(parent_node.hd_message.T).to(device)
                    guided_layers[self.n_layer-layer][:,position,self.variable_type:] = torch.tensor(parent_node.qd_message.T).to(device)

                goal_node_parents.append(goal_node_parents[-1].parent)
                id_goal_nodes.append(id_goal_nodes[-1]//self.n_child)
                node = parent_node
                id_node = parent_id_node # update the id node
                if node == goal_node_parents[-1]:
                    share_parents_flags.append(True)
                else:
                    share_parents_flags.append(False)
            
            # Update the root node
            # self.root_node.hd_message += node.qd_message # update the hd message
            self.root_node.hd_message = 0
            for id_child in range(self.n_child): # update the hd message
                if id_child  <= id_node:
                    self.root_node.hd_message += self.root_node.children[id_child].qd_message
            self.root_node.hd_message -= self.root_node.hd_message.max(0) # normalize the hd message
            self.root_node.bu_message = self.root_node.hd_message # update the bu message

            # information from the external source
            if external_hd_message is not None:
                self.root_node.bu_message += external_hd_message
            if verbose and position == pos:
                print("Root", self.root_node.hd_message.T)
                print("Root", self.root_node.bu_message.T)
                print("Upward Process")
            
            self.root_node.bu_message -= self.root_node.bu_message.max(0) # normalize the bu message

            if guide_info: # save the guided information for the root node
                guided_layers[self.n_layer][:,position,:self.variable_type] = torch.tensor(self.root_node.hd_message.T).to(device)
                guided_layers[self.n_layer][:, position,self.variable_type:] = torch.tensor(self.root_node.bu_message.T).to(device)

            # upward process
            for layer in range(1, self.n_layer+1):
                node = goal_node_parents[-layer]
                if share_parents_flags[-layer]: # if the current node and the goal node share the same parent
                    diff = node.parent.bu_message - node.qd_message # difference between the parent bu message and the qd message
                    # print( self.transition[layer-1][id_goal_nodes[-layer]])
                    # update = self.transition[layer-1][id_goal_nodes[-layer]].T @ np.exp(diff) # update for the bu message
                    # if np.sum(update ==0) > 0:
                    #     print(update)
                    node.bu_message = node.hd_message + np.log(self.transition[layer-1][id_goal_nodes[-layer]].T @ np.exp(diff)) # update the bu message
                else:
                    node.bu_message = np.log(self.transition[layer-1][id_goal_nodes[-layer]].T @ np.exp(node.parent.bu_message))
                node.bu_message -= node.bu_message.max(0) # normalize the bu message
                if verbose and position == pos:
                    print("Layer", self.n_layer - layer, node.bu_message.T)

                if guide_info: # save the guided information for the current layer
                    guided_layers[self.n_layer+layer][:,position,:] = torch.tensor(node.bu_message.T).to(device)
            
            predict_pp[:,position,:] = torch.tensor(np.exp(node.bu_message)/np.sum(np.exp(node.bu_message),0)).to(device).T # posterior probability for the next word
        
        return  predict_pp, guided_layers



    def BP_DNS(self, z, sigma=1.0, external_hd_message=None):
        """
        Belief propagation for denoising tasks. 
        """
        self.posterior_mean_DNS = np.zeros(z.shape)
        
        # generate the true transtion probability
        p_flip = self.p_flip/(self.variable_type ) 
        p_same = 1- p_flip * (self.variable_type - 1) 

        # generate message for the leaf nodes 
        vt_choices = np.linspace(0, self.variable_type - 1, self.variable_type) # possible values for the variable
        index = 0
        for id_node in range(len(self.Tree[-1])):
            node = self.Tree[-1][id_node]
            node.hd_message = -0.5 * (z[index,:] - vt_choices.reshape([self.variable_type, 1])) ** 2 / (sigma ** 2)
            node.qd_message = np.log(self.transition[-1][id_node] @ np.exp(node.hd_message))
            index += 1

        # downward process 
        for layer in range(self.n_layer-1, 0, -1):
            for id_node in range(len(self.Tree[layer])):
                node = self.Tree[layer][id_node]
                node.hd_message = sum( child.qd_message 
                                                for child in node.children)
                node.hd_message -= np.max(node.hd_message, 0)
                node.qd_message = np.log(self.transition[layer-1][id_node] @ np.exp(node.hd_message)) 
        
        # Update the root node 
        self.root_node.hd_message = sum( child.qd_message 
                                                for child in self.root_node.children)
        self.root_node.hd_message -= np.max(self.root_node.hd_message, 0)
        self.root_node.bu_message = self.root_node.hd_message 
        if external_hd_message is not None:
            self.root_node.bu_message += external_hd_message

        # upward process
        for layer in range(1, self.n_layer+1):
            for id_node in range(len(self.Tree[layer])):
                node = self.Tree[layer][id_node]
                diff = node.parent.bu_message - node.qd_message
                node.bu_message = node.hd_message + np.log(self.transition[layer-1][id_node].T @ np.exp(diff))
                node.bu_message -= np.max(node.bu_message, 0)
        # posterior mean 
        index = 0
        for node in self.leaves_nodes:
            self.posterior_mean_DNS[ index,:] = vt_choices.dot(np.exp(node.bu_message)) / np.sum(np.exp(node.bu_message),0)
            index += 1
        
        self.dns_flag = True

        return self.posterior_mean_DNS        

    
    def guided_info(self, device='cpu'):
        """
        Guided information for the tree. 
        """
        
        guided_layers = []
        extend_time = 1
        if self.cls_flag:            

            for i in range(self.n_layer-1,-1,-1):
                extend_time *= self.n_child
                guided_layer = [] 

                for node in self.Tree[i]:
                    if node.hd_message is not None:
                        guided_layer.extend([node.hd_message.copy() for _ in range(extend_time)])

                guided_layer = torch.tensor(np.array(guided_layer), dtype=torch.float)
                guided_layer = torch.transpose(guided_layer, 0, 1)
                guided_layer = torch.transpose(guided_layer, 0, 2)
                guided_layer = guided_layer.to(device)
                guided_layers.append(guided_layer)
        
        elif self.dns_flag:

            for i in range(self.n_layer,0,-1):
                guided_layer_h = []
                guided_layer_q = [] 
                for node in self.Tree[i]:
                    if node.hd_message is not None:
                        guided_layer_h.extend([node.hd_message.copy() for _ in range(extend_time)])
                        guided_layer_q.extend([node.qd_message.copy() for _ in range(extend_time)])
                guided_layer = torch.tensor(np.hstack([np.array(guided_layer_h), np.array(guided_layer_q)]), dtype=torch.float)
                guided_layer = torch.transpose(guided_layer, 0, 1)
                guided_layer = torch.transpose(guided_layer, 0, 2)
                guided_layer = guided_layer.to(device)
                guided_layers.append(guided_layer)
                extend_time *= self.n_child

            guided_layer_h = [self.root_node.hd_message.copy() for _ in range(extend_time)]
            guided_layer_q = [self.root_node.bu_message.copy() for _ in range(extend_time)]
            guided_layer = torch.tensor(np.hstack([np.array(guided_layer_h), np.array(guided_layer_q)]), dtype=torch.float)
            guided_layer = torch.transpose(guided_layer, 0, 1)
            guided_layer = torch.transpose(guided_layer, 0, 2)
            guided_layer = guided_layer.to(device)
            guided_layers.append(guided_layer)            
            
            for i in range(self.n_layer):
                extend_time = extend_time // self.n_child
                guided_layer_h = []
                guided_layer_q = []
                guided_layer_b = []
                for node in self.Tree[i+1]:
                    if node.hd_message is not None:
                        guided_layer_h.extend([node.hd_message.copy() for _ in range(extend_time)])
                        guided_layer_q.extend([node.qd_message.copy() for _ in range(extend_time)])
                        guided_layer_b.extend([node.bu_message.copy() for _ in range(extend_time)])
                guided_layer = torch.tensor(np.hstack([np.array(guided_layer_h), np.array(guided_layer_q), np.array(guided_layer_b)]), dtype=torch.float)
                guided_layer = torch.transpose(guided_layer, 0, 1)
                guided_layer = torch.transpose(guided_layer, 0, 2)
                guided_layer = guided_layer.to(device)
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
        return self.T_value[-1]

    @property
    def root_value(self):
        return self.T_value[0][0]

# Sampler Classes

class SingleSampler:
    """
    Single sampler for the GHM tree. 
    """
    def __init__(self, n_layer, n_child, p_y, p_flip,flip_scale=1.0, variable_type=10, translation_invariance=True, seedtree=42):
        self.n_layer = n_layer
        self.n_child = n_child
        self.p_y = p_y
        self.p_flip = p_flip
        self.variable_type = variable_type
        self.translation_invariance = translation_invariance
        self.seedtree = seedtree
        self.flip_scale = flip_scale

        np.random.seed(seedtree)

        self.transition = GenTransition(n_layer, n_child, variable_type, p_flip, flip_scale,
        translation_invariance=translation_invariance)
    
    def get_batch(self, batch_size=128):
        T = GHMTree(self.n_layer, self.n_child, self.variable_type, self.p_y, self.p_flip, self.transition, batch_size, build_tree=True)
        return T.T_value[0][0], T.T_value[-1][0]

class DoubleSampler:
    """
    Double sampler for the GHM tree. One is the image sampler and the other is the text sampler. 
    """
    def __init__(self, n_layers, n_childs, p_ys, p_flips, flip_scale=1, variable_type = 10, translation_invariance=True, seedtree = 42):
        self.n_layers = n_layers
        self.n_childs = n_childs
        self.p_ys= p_ys
        self.p_flips = p_flips 
        self.flip_scale = flip_scale
        self.variable_type = variable_type
        self.seedtree = seedtree

        np.random.seed(seedtree)

        self.t_transition = GenTransition(n_layers[0], n_childs[0], variable_type, p_flips[0], flip_scale, translation_invariance=translation_invariance)

        self.i_transition = GenTransition(n_layers[1], n_childs[1], variable_type, p_flips[1], flip_scale, translation_invariance=translation_invariance)
    
    def get_batch(self, batch_size=128):
        text_tree = GHMTree(self.n_layers[0], self.n_childs[0], self.variable_type, self.p_ys[0], self.p_flips[0], self.t_transition, batch_size, build_tree=True)
        image_tree = GHMTree(self.n_layers[1], self.n_childs[1], self.variable_type, self.p_ys[1], self.p_flips[1], self.i_transition, batch_size, build_tree=True)

        return text_tree.T_value[0][0], image_tree.T_value[0][0], text_tree.T_value[-1][0], image_tree.T_value[-1][0]
    
    def get_zeroshot_batch(self, batch_size=128, return_tree=False):
        root_tree = np.random.choice(self.variable_type, size = batch_size)
        text_tree = GHMTree(self.n_layers[0], self.n_childs[0], self.variable_type, self.p_ys[0], self.p_flips[0], self.t_transition, batch_size, build_tree=True, root=root_tree)
        image_tree = GHMTree(self.n_layers[1], self.n_childs[1], self.variable_type, self.p_ys[1], self.p_flips[1], self.i_transition, batch_size, build_tree=True, root=root_tree)
        text_tree.BP_CLS()
        image_tree.BP_CLS()
        if return_tree:
            return text_tree, image_tree 
        else:
            return np.array(text_tree.leaves_values).T, np.array(image_tree.leaves_values).T, np.array(text_tree.posterior_probability_CLS).T, np.array(image_tree.posterior_probability_CLS).T, np.array(root_tree)

# Single Sampler Class
class ClassificationSampler(SingleSampler):
    def __init__(self, n_layer, n_child, p_y, p_flip=0.3, flip_scale=1, variable_type=10,translation_invariance=True, seedtree=42):

        super().__init__(n_layer, n_child, p_y, p_flip,flip_scale, variable_type, translation_invariance, seedtree) 

    def get_batch(self, batch_size=128, guide=False, device="cpu"):
        tree = GHMTree(self.n_layer, self.n_child, self.variable_type, self.p_y, self.p_flip, self.transition, batch_size, build_tree=True)
        leaves_values = torch.tensor(tree.leaves_values, dtype=torch.long).T.to(device)
        root_values = torch.tensor(tree.root_value, dtype=torch.long).to(device)
        if guide:
            tree.BP_CLS()
            guided_info = tree.guided_info(device=device)
        else:
            guided_info = None 
        
        return leaves_values, root_values, guided_info, tree.posterior_probability_CLS.T
    
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

    def __init__(self, n_layer, n_child, p_y, p_flip=0.3, sigma=1, flip_scale=1, variable_type=10,translation_invariance=True, seedtree=42):

        super().__init__(n_layer, n_child, p_y, p_flip,flip_scale, variable_type, translation_invariance, seedtree)
        self.sigma = sigma
    
    def get_batch(self, batch_size=128, guide=False, device="cpu"):
        tree = GHMTree(self.n_layer, self.n_child, self.variable_type, self.p_y, self.p_flip, self.transition, batch_size, build_tree=True)
        zs = np.random.randn(self.n_child**self.n_layer, batch_size) * self.sigma + tree.leaves_values
        xs = torch.tensor(tree.leaves_values, dtype=torch.float).T.to(device)
        if guide:
            tree.BP_DNS(zs, self.sigma)
            guided_info = tree.guided_info()
        else:
            guided_info = None 
        
        zs = torch.tensor(zs, dtype=torch.float).T.to(device)
        return zs, xs, guided_info, tree.posterior_mean_DNS.T


# Double Sampler Class
class ClipSampler(DoubleSampler):

    def __init__(self, n_layers, n_childs, p_ys, p_flips, K=4, flip_scale=1, variable_type=10, translation_invariance=True, seedtree=42):
        super().__init__(n_layers, n_childs, p_ys, p_flips, flip_scale, variable_type, translation_invariance, seedtree)
        self.K = K
    
    def get_batch(self, device="cpu", batch_size=128, guide=False):
            text_tree_root = np.random.choice(self.variable_type, size = batch_size *(self.K+1)) 
            image_tree_root = np.random.choice(self.variable_type, size = batch_size *(self.K-1)) 
            image_tree_root = np.append(text_tree_root[:2*batch_size], image_tree_root) 

            text_tree = GHMTree(self.n_layers[0], self.n_childs[0], self.variable_type, self.p_ys[0], self.p_flips[0], self.t_transition, batch_size*(self.K+1), build_tree=True, root=text_tree_root)

            image_tree = GHMTree(self.n_layers[1], self.n_childs[1], self.variable_type, self.p_ys[1], self.p_flips[1], self.i_transition, batch_size*(self.K+1), build_tree=True, root=image_tree_root)

            if guide:
                text_tree.BP_CLS()
                image_tree.BP_CLS()
                text_guided_info = text_tree.guided_info(device=device)
                image_guided_info = image_tree.guided_info(device=device)
                t_pp = text_tree.posterior_probability_CLS.T
                i_pp = image_tree.posterior_probability_CLS.T
            else:
                text_guided_info = None
                image_guided_info = None
                t_pp = None
                i_pp = None
            
            image_tree_leaves = torch.tensor(image_tree.leaves_values, dtype=torch.long).T.to(device)
            text_tree_leaves = torch.tensor(text_tree.leaves_values, dtype=torch.long).T.to(device)
            image_tree_root = torch.tensor(image_tree_root, dtype=torch.long).to(device)
            text_tree_root = torch.tensor(text_tree_root, dtype=torch.long).to(device)

            return [text_tree_leaves, text_tree_root, text_guided_info, t_pp], [image_tree_leaves, image_tree_root, image_guided_info, i_pp]
    
    def get_Bayes(self,n_eval=10000):
        """
        Get the Bayesian Error.
        """
        res = self.get_batch(batch_size=n_eval, guide=True)
        ttree_pp = res[0][3].T
        itree_pp = res[1][3].T

        # K imdage and 1 text 
        t_pp_match = ttree_pp[:, :n_eval] 
        i_pp_match = itree_pp[:, :n_eval] 
        t_pp_indep = ttree_pp[:, 2*n_eval:] 

        # compute S 
        S_match = np.sum(t_pp_match * i_pp_match, 0) *self.variable_type
        S_indep = np.sum(t_pp_indep * np.tile(i_pp_match, (1,self.K-1)),0)
        concat_mat = np.kron(np.ones([self.K-1,1]), np.eye(n_eval))
        S_indep = S_indep.dot(concat_mat)*self.variable_type 
        S = -np.log(S_match/(S_indep + S_match)) 

        # K text and 1 image 
        t_pp_match = ttree_pp[:, n_eval: 2*n_eval] 
        i_pp_match = itree_pp[:, n_eval: 2*n_eval]
        i_pp_indep = itree_pp[:, 2*n_eval:] 

        # compute S 
        S_match = np.sum(t_pp_match * i_pp_match, 0) *self.variable_type
        S_indep = np.sum(i_pp_indep * np.tile(t_pp_match, (1,self.K-1)),0)
        S_indep = S_indep.dot(concat_mat)*self.variable_type 
        S += -np.log(S_match/(S_indep + S_match)) 

        return np.mean(S), np.std(S) / np.sqrt(n_eval)

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
        
        return (input_text_tree_sequence, target_text_tree_sequence, text_guided_info, BP_output), (image_tree_leaves, image_tree_roots, image_guided_info, image_tree.posterior_probability_CLS.T)

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
        return torch.mean(loss), torch.std(loss) / np.sqrt(n_eval)





if __name__ == "__main__":
    n_layer = 4
    n_child = 3
    p_y = np.ones(10) / 10
    p_flip = 0.3
    sigma = 2
    batch_size =2
    # variable_type  = "binary"
    variable_type = 10

    n_layers = [3,4]
    n_childs = [3,3]
    p_ys = [np.ones(10)/10, np.ones(10)/10]
    p_flips = [0.1, 0.1]
    K = 4
    variable_type = 10
    flip_scale = 1
    translation_invariance = True
    sigma=0.1
    
    sampler = ClassificationSampler(n_layer, n_child, p_y, p_flip, variable_type=variable_type, translation_invariance=True)
    print(sampler.get_Bayes())

    # sampler = ClipSampler(n_layers, n_childs, p_ys, p_flips, K, flip_scale, variable_type, translation_invariance)
    # print(sampler.get_Bayes())
    # transition = GenTransition(n_layer, n_child, variable_type, p_flip, translation_invariance=True)
    # tree= GHMTree(n_layer, n_child, variable_type, p_y, p_flip,transition=transition,batch_size=batch_size, build_tree=True)
    # ind = 3
    # print(np.mean(tree.leaves_values[ind]))
    # print(np.mean(np.power(tree.leaves_values[ind],2)))
    # variable_choices = np.linspace(0, variable_type-1, variable_type).reshape([1, variable_type])
    # print(np.mean(variable_choices.dot(tree.BP_NWP(ind))))
    # print(np.mean(np.power(variable_choices,2).dot(tree.BP_NWP(ind))))
    # true_values = tree.leaves_values[ind]
    # print(tree.leaves_values[ind])
    # print(tree.BP_NWP(ind))
    # print(tree.BP_dummy_NWP(ind))
    # predict_pp, guided_layers = tree.BP_NWP_autoregressive(guide_info=True, verbose=False)
    # print(predict_pp[:,1,:])
    # print(tree.leaves_values[2])
    # print(tree.BP_dummy_NWP(2).T)
    # print(tree.BP_NWP(2))

    # print(prediction.shape)
    # for guided_layer in guided_layers:
    #     print(guided_layer.shape)

    # sampler = NextWordPredictSampler([3,4], 
    #                                  [3,3], 
    #                                  [np.ones(10)/10, np.ones(10)/10], 
    #                                  [0.1, 0.1], 
    #                                  variable_type=10)

    # n_ttree_layer = 4
    # n_itree_layer = 4
    # n_ttree_child = 3
    # n_itree_child = 3
    # p_ttree_flip = 0.4
    # p_itree_flip = 0.4
    # flip_scale = 1
    # p_y = np.ones(10) / 10
    # guide = True 

    # device = "cpu"
    # sampler = NextWordPredictSampler([n_ttree_layer, n_itree_layer], 
    #                             [n_ttree_child, n_itree_child], 
    #                             [p_y, p_y], 
    #                             [p_ttree_flip, p_itree_flip], 
    #                             flip_scale=flip_scale,
    #                             variable_type=variable_type, 
    #                             translation_invariance=True, 
    #                             seedtree=42)
    # a,b = sampler.get_Bayes()
    # print(a,b)

    # np.random.seed(42)
    # p_flip = 0.2
    # transitions, skeletons = GenTransition(n_layer, n_child, variable_type, p_flip, flip_scale, translation_invariance=True, verbose=True)
    # print(skeletons[0])
    # print("+++++++++++++++++++++++++++++")
    # p_flip = 0.4
    # np.random.seed(42)
    # transitions, skeletons = GenTransition(n_layer, n_child, variable_type, p_flip, flip_scale, translation_invariance
    # =True, verbose=True)
    # print(skeletons[0])