import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemN2N(nn.Module):
    """End-2-End Memory Network."""
 
    def __init__(self, 
                 mem_cell_size, 
                 vocab_size, 
                 candidate_vecs, 
                 n_hops,
                 init_std=0.1,
                 nonlinearity=True):
        """
        :param mem_cell_size: Size of the memory cell.
        :param vocab_size: Total number words in the vocabulary.
        :param candidate_vecs: Tensor containing vectors (vector of word indices) for each candidate response.
        :param n_hops: Number of iterative memory accesses.
        :param init_std: Initial std for weight initialization.
        :param nonlinearity: If true, performs softmax normalization of attention weights.
        """
        
        super(MemN2N, self).__init__()
        
        self.mem_cell_size = mem_cell_size
        self.vocab_size = vocab_size
        self.candidate_vecs = candidate_vecs
        self.n_hops = n_hops
        self.init_std = init_std
        self.nonlinearity = nonlinearity
        
        self.query_emb = nn.Embedding(vocab_size, mem_cell_size, padding_idx=0)
        self.query_emb.weight.data.normal_(std=init_std)
        self.query_emb.weight.data[0] = 0
        
        self.out_transform = nn.Linear(mem_cell_size, mem_cell_size, bias=False)
        self.out_transform.weight.data.normal_(std=init_std)
        
        self.mem_emb = nn.ModuleList()
        for i in range(n_hops):
            
            mem_emb = nn.Embedding(vocab_size, mem_cell_size, padding_idx=0)
            mem_emb.weight.data.normal_(std=init_std)
            mem_emb.weight.data[0] = 0
            
            self.mem_emb.append(mem_emb)
            
        self.candidate_emb = nn.Embedding(vocab_size, mem_cell_size, padding_idx=0)
        self.candidate_emb.weight.data.normal_(std=init_std)
        self.candidate_emb.weight.data[0] = 0

    def forward(self, memory, query):
        """
        :param memory: torch Variable, containing memory vectors.
        :param query: torch Variable, containing query vector.
        :return: Pair of log softmax predictions and attention layer activations.
        """
        
        query_emb = torch.sum(self.query_emb(query), dim=1)
        
        u = [query_emb]
        attns = []
        for hop in range(self.n_hops):       
            
            mem_emb = self.embed_memory(memory, hop)

            attn_weights = torch.bmm(mem_emb, u[-1].unsqueeze(2))

            if self.nonlinearity:
                attn_weights = F.softmax(attn_weights, 1)

            output_tmp = torch.bmm(attn_weights.permute(0, 2, 1), mem_emb).squeeze(1)
            output = self.out_transform(output_tmp)
            
            u.append(u[-1] + output)
            attns.append(attn_weights)
            
        candidate_emb = torch.sum(self.candidate_emb(self.candidate_vecs), dim=1)
        
        y_pred = candidate_emb @ u[-1].permute(1, 0)
        
        return F.log_softmax(y_pred, dim=0).permute(1, 0), torch.stack(attns, 3).squeeze(2)

    def embed_memory(self, memory, hop):

        emb = self.mem_emb[hop](memory.view(-1, memory.size()[2]))

        emb = torch.sum(emb.view(*memory.size(), -1), dim=2)
        
        return emb

    def save(self, path=None):
        """
        Saves model state so that it can be restored later.

        :param path: Path to the save directory.
        """

        if not path:
            path = os.path.join(os.getcwd(), 'model_' + str(time.time()))
        
        torch.save({
            'mem_cell_size': self.mem_cell_size,
            'vocab_size': self.vocab_size,
            'candidate_vecs': self.candidate_vecs,
            'n_hops': self.n_hops,
            'init_std': self.init_std,
            'nonlinearity': self.nonlinearity,
            'state_dict': self.state_dict()
        }, path)

    @staticmethod    
    def load(path, load_weights=True):
        """
        Static factory that builds the previously stored model.

        :param path: Path to the saved model.
        :param load_weights: If False, model weights (learnable parameters) aren't restored.
        """

        model_params = torch.load(path, map_location=lambda storage, loc: storage.cpu())
        
        model = MemN2N(model_params['mem_cell_size'], 
                       model_params['vocab_size'],
                       model_params['candidate_vecs'],
                       model_params['n_hops'],
                       model_params['init_std'],
                       model_params['nonlinearity'])
        
        if load_weights:
            model.load_state_dict(model_params['state_dict'])
        
        return model
