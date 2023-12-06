import copy
import pdb
import numpy as np
import time
from pdb import set_trace
# Import the optimization libraries
import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple

import logging as lg
lg.basicConfig(level=lg.DEBUG)

class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
 
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
 
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
 
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
 
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
 
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
 
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)
 
    def forward(self, x):
        return self.norm(x)

class CustomDataset(Dataset):
    def __init__(self, metadata, metadata_prev, gt):
        super(CustomDataset, self).__init__()
        self.metadata = metadata
        self.metadata_prev = metadata_prev
        self.gt = gt
        
    def __len__(self):
        return self.metadata.shape[0]
    
    def __getitem__(self, idx):        
        return self.metadata[idx, :], self.metadata_prev[idx, :], self.gt[idx]

class TransformerDecoder(nn.Module):
    def __init__(self, n_cls, n_clfs_prev, n_clfs, embedding_dim=64, ff_dim=64, dropout=0.8):
        super(TransformerDecoder, self).__init__()
        
        self.n_cls = n_cls
        self.n_clfs_prev = n_clfs_prev
        self.n_clfs = n_clfs        
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim
        
        self.ff_prev = nn.Linear(n_cls, embedding_dim)
        self.ff_curr = nn.Linear(n_cls, embedding_dim)        
        self.encoder_attention = ScaledDotProductAttention(embedding_dim)
        # self.ff_temp = nn.Linear(n_clfs*(n_clfs_prev+n_cls), embedding_dim)
        # self.ff_temp = nn.Linear(n_clfs*(n_clfs_prev+embedding_dim), embedding_dim)
        self.ff_temp = nn.Linear(n_clfs*(n_clfs_prev+embedding_dim), n_cls)
        
    def forward(self, metadata, metadata_prev):
        # metadata_prev_ff = self.ff_prev(metadata_prev)
        # metadata_ff = self.ff_curr(metadata)        
        # context, attn = self.encoder_attention(metadata_ff, metadata_prev_ff, metadata_prev_ff)        
        # x1 = metadata_ff + context
        # x1 = torch.cat((context, attn), dim=2)
        # x1 = x1.view(-1, self.n_clfs*(self.n_cls+self.n_clfs_prev))
        # x4 = self.ff_temp(x1)
        
        metadata_prev_ff = self.ff_prev(metadata_prev)
        metadata_ff = self.ff_curr(metadata)        
        context, attn = self.encoder_attention(metadata_ff, metadata_prev_ff, metadata_prev_ff)        
        x1 = metadata_ff + context
        x1 = torch.cat((context, attn), dim=2)
        x1 = x1.view(-1, self.n_clfs*(self.embedding_dim+self.n_clfs_prev))
        x4 = self.ff_temp(x1)
        # set_trace()
        return (x4, attn)
      
def run_transformer_decoder(metadata_prev, metadata, gt, n_cls, n_clfs_prev, n_clfs, n_epoch=500, batch_size=128):
    """ Run the attention-based module for ensemble selection

    Parameters
    ----------
    metadata_prev : Numpy array.   
        The meta-data of the previous layer, with size (n_inst, n_clfs_prev, n_cls), where
        n_inst is the number of instances, n_clfs_prev is the number of chosen classifiers in the 
        previous layer, and n_cls is the number of classes.
    metadata : Numpy array.   
        The meta-data of the current layer, with size (n_inst, n_clfs, n_cls), where
        n_inst is the number of instances, n_clfs is the number of all classifiers in the 
        current layer (we run with all classifiers chosen first to get this array), and
        n_cls is the number of classes.    
    gt: Numpy array.
        The ground truth, of size (n_inst, 1).
    n_cls: Integer.
        The number of classes.
    n_clfs_prev: Integer.
        The number of chosen classifiers in the previous layer.
    n_clfs: Integer.
        The total number of classifiers in the current layer.
    n_epoch: Integer, optional.
        The number of epochs to run for training (default is 500).
    batch_size: Integer, optional.
        The batch size (default is 128).    

    Returns
    -------
    attn: Numpy array.
        The attention matrix which shows the attention coefficients between the classifiers 
        of the previous and current layer in the ensemble, with size (n_inst, n_clfs, n_clfs_prev).
    train_loss_list: List of floats.
        The training loss.
    val_loss_list: List of floats.
        The validation loss.
    """

    n_inst = metadata_prev.shape[0]
    metadata_prev_rs = metadata_prev.reshape((n_inst, n_clfs_prev, n_cls))
    metadata_rs = metadata.reshape((n_inst, n_clfs, n_cls))
    
    DEVICE = "cpu"
    
    metadata_prev_torch = torch.from_numpy(metadata_prev_rs).to(DEVICE, dtype=float)
    metadata_torch = torch.from_numpy(metadata_rs).to(DEVICE, dtype=float)
    gt_torch = torch.from_numpy(gt).to(DEVICE, dtype=float)

    # model = TransformerDecoder(n_cls, n_clfs, embedding_dim=n_cls).to(DEVICE, dtype=float)
    model = TransformerDecoder(n_cls, n_clfs_prev, n_clfs, embedding_dim=64, ff_dim=64).to(DEVICE, dtype=float)
    # model = TransformerDecoderMultiHead(n_cls, n_clfs, embedding_dim=64, ff_dim=64).to(DEVICE, dtype=float)
    # (output, attn) = model(metadata_torch, metadata_prev_torch)
    
    # Train
    # n_epoch = 10000
    # n_epoch = 500
    # n_epoch = 10
    # batch_size = 128
    # batch_size = 392
    
    train_dataset = CustomDataset(metadata_torch, metadata_prev_torch, gt_torch)    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    # optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3)
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    # lambda1, lambda2 = 0.5, 0.01
    output_hook = OutputHook()
    model.encoder_attention.register_forward_hook(output_hook)
    l1_lambda = 0.01
    

    # scale = 0.09
    scale = 1/(1.0*n_clfs)
    # scale = 1.0
    coeff = 0.01

    best_loss = 1000000
    best_model = None
    train_loss_list = []
    val_loss_list = []    
    
    for epoch_idx in range(1, n_epoch+1):
        model.train()
        mini_batch_losses = []
        for (metadata_torch_batch, metadata_prev_torch_batch, gt_torch_batch) in train_dataloader:
            optimizer.zero_grad()
            
            (output_batch, attn) = model(metadata_torch_batch, metadata_prev_torch_batch)            
            gt_torch_batch = gt_torch_batch.long()
            gt_torch_batch -= 1 # 1, 2, 3, ... -> 0, 1, 2, ...
       
            # Attention elimination term, as described in C. Bishop, Neural networks for Pattern recognition, 1995, p. 363.
            attn_squared = torch.square(attn)
            the_penalty = torch.sum(torch.div(attn_squared, scale*scale+attn_squared))
            the_penalty /= batch_size
            loss = loss_fn(output_batch, gt_torch_batch) + coeff*the_penalty
            
            # L1 regularization term applied on the attention matrix
            # attn_abs = torch.abs(attn)
            # l1_penalty = torch.sum(attn_abs)/batch_size
            # loss = loss_fn(output_batch, gt_torch_batch) + l1_lambda*l1_penalty
            
            # l1_penalty *= l1_lambda
            
            # l1_penalty = l1_lambda * sum([p.abs().sum() for p in model.encoder_attention.parameters()])
            
            # loss = loss_fn(output_batch, gt_torch_batch)
            # loss = loss_fn(output_batch, gt_torch_batch) + l1_penalty
            
            loss.backward()
            optimizer.step()
            output_hook.clear()
            mini_batch_losses.append(loss.item())
        
        train_loss = np.mean(mini_batch_losses)        
        
        model.eval()
        mini_batch_losses = []
        with torch.no_grad():
            for (metadata_torch_batch, metadata_prev_torch_batch, gt_torch_batch) in val_dataloader:
                (output_batch, attn) = model(metadata_torch_batch, metadata_prev_torch_batch)
                gt_torch_batch = gt_torch_batch.long()
                gt_torch_batch -= 1 # 1, 2, 3, ... -> 0, 1, 2, ...
                
                # Attention elimination term, as described in C. Bishop, Neural networks for Pattern recognition, 1995, p. 363.
                attn_squared = torch.square(attn)                
                the_penalty = torch.sum(torch.div(attn_squared, scale*scale+attn_squared))
                the_penalty /= batch_size
                loss = loss_fn(output_batch, gt_torch_batch) + coeff*the_penalty
                
                # L1 regularization term applied on the attention matrix
                # attn_abs = torch.abs(attn)
                # l1_penalty = torch.sum(attn_abs)/batch_size
                # loss = loss_fn(output_batch, gt_torch_batch) + l1_lambda*l1_penalty
                
                # loss = loss_fn(output_batch, gt_torch_batch)
                
                # set_trace()
                mini_batch_losses.append(loss.item())
        val_loss = np.mean(mini_batch_losses)
        
        print("Epoch %d. Train loss: %f. Val loss: %f" % (epoch_idx, train_loss, val_loss))
        
        if val_loss < best_loss:            
            best_model = copy.deepcopy(model.state_dict())
            best_loss = val_loss
            print("Saving best model at epoch %d with best loss so far: %f" % (epoch_idx, best_loss))

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
            
    # Return the attention scores
    model.load_state_dict(best_model)
    (output, attn) = model(metadata_torch, metadata_prev_torch)
    attn = attn.cpu().detach().numpy()    
    return attn, train_loss_list, val_loss_list
    
def test_1():
    print("")
    print("")
    print("")
    print("test_1")
    
    n_inst = 11
    n_clfs_prev = 4    
    n_clfs = 7
    n_cls = 3
    
    metadata_prev = torch.rand((n_inst, n_clfs_prev, n_cls))
    metadata = torch.rand((n_inst, n_clfs, n_cls))
    attn_module = ScaledDotProductAttention(1)
    context, attn = attn_module(metadata_prev, metadata, metadata)
    
    print("n_inst = %d" % (n_inst))
    print("n_clfs_prev = %d" % (n_clfs_prev))
    print("n_clfs = %d" % (n_clfs))
    print("n_cls = %d" % (n_cls))
    
    print(metadata_prev.shape)
    print(metadata.shape)
    print(context.shape)
    print(attn.shape)

def test_2():
    print("")
    print("")
    print("")
    print("test_2")
    
    n_inst = 11
    n_clfs_prev = 4    
    n_clfs = 7
    n_cls = 3
    
    DEVICE = "cpu"
    
    metadata_prev_torch_batch = torch.rand((n_inst, n_clfs_prev, n_cls)).to(DEVICE, dtype=float)
    metadata_torch_batch = torch.rand((n_inst, n_clfs, n_cls)).to(DEVICE, dtype=float)
    
    model = TransformerDecoder(n_cls, n_clfs_prev, n_clfs, embedding_dim=64, ff_dim=64).to(DEVICE, dtype=float)
    (output_batch, attn) = model(metadata_torch_batch, metadata_prev_torch_batch)
    
    print("n_inst = %d" % (n_inst))
    print("n_clfs_prev = %d" % (n_clfs_prev))
    print("n_clfs = %d" % (n_clfs))
    print("n_cls = %d" % (n_cls))
    
    print("output_batch.shape = %s" % (output_batch.shape,))
    print("attn.shape = %s" % (attn.shape,))

def test_3():
    print("")
    print("")
    print("")
    print("test_3")
    
    n_inst = 20
    n_clfs_prev = 4    
    n_clfs = 7
    n_cls = 3
    n_epoch = 3
    batch_size = 4
    
    metadata_prev = np.random.rand(n_inst, n_clfs_prev, n_cls)
    metadata = np.random.rand(n_inst, n_clfs, n_cls)
    gt = np.ones((n_inst), dtype=np.int32)    
    attn, train_loss_list, val_loss_list = run_transformer_decoder(metadata_prev, metadata, gt, n_cls, n_clfs_prev, n_clfs,
        n_epoch=n_epoch, batch_size=batch_size)
        
    print("n_inst = %d" % (n_inst))
    print("n_clfs_prev = %d" % (n_clfs_prev))
    print("n_clfs = %d" % (n_clfs))
    print("n_cls = %d" % (n_cls))
    print("attn.shape = %s" % (attn.shape,))    
    
def main():
    test_1()
    test_2()
    test_3()
    
if __name__ == '__main__':
    main()
