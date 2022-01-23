# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor


class AttentiveLinear(nn.Module):
    def __init__(self,
                 embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings # [|R|,k]; E_R in the paper: predicate embedding matrix
        nb_objects = self.embeddings.weight.shape[0] # |R| in the paper: number of available relations
        embedding_size = self.embeddings.weight.shape[1] # k in the paper
        self.projection = nn.Linear(embedding_size, nb_objects) # [k,|R|]; W_i in the paper

    def forward(self, rel: Tensor) -> Tensor:
        # [B, O]
        attn_logits = self.projection(rel) #W_i * x in the paper
        attn = torch.softmax(attn_logits, dim=1) # alpha in the paper
        # [B, E]
        return attn @ self.embeddings.weight
