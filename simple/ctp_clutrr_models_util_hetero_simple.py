# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.kernels import BaseKernel
from ctp.kernels import GaussianKernel

from typing import Tuple, Optional


# gives back the similarity measures of each fact in facts to the corresponding query rule
# size [B,F], but where there's less than F facts the rest are zeroed out (similarity is 0)
def lookup(query: Tensor, # batch_emb [B,3E]: B: batches, E: embedding size
           facts: Tensor, # facts_emb [F,3E]: F: maximum number of facts
           kernel: BaseKernel) -> Tensor:
    # query: [B, E], facts: [B, F, E], nb_facts: [B]
    batch_size, fact_size = query.shape[0], facts.shape[0]

    #facts, nb_facts = uniform(query, facts, nb_facts)
    #facts, _ = uniform(query, facts)

    #assert query.shape[0] == facts.shape[0]  # == nb_facts.shape[0]
    #assert query.shape[1] == facts.shape[2]

    # similarity measure between the query embeddings and the fact embeddings
    kernel_values = kernel(query, facts).view(batch_size, fact_size)  # [B,F]

    return kernel_values

# if we have more query fact embeddings (a) in our batch than the batch size of our fact embeddings (b) then we repeat some
# fact embeddings to have the same instances in the two batches (only works well if we have more by an integer multiple)
def uniform(a: Tensor,
            b: Tensor,
            c: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
    if a.shape[0] > b.shape[0]:
        m = a.shape[0] // b.shape[0]
        b = b.view(b.shape[0], 1, b.shape[1], b.shape[2]).repeat(1, m, 1, 1).view(-1, b.shape[1], b.shape[2])
        if c is not None:
            c = c.view(-1, 1).repeat(1, m).view(-1)
    return b, c


if __name__ == '__main__':
    kernel = GaussianKernel()

    batch_size = 8
    fact_size = 32
    embedding_size = 10

    query = torch.rand(batch_size, embedding_size)
    facts = torch.rand(batch_size, fact_size, embedding_size)
    nb_facts = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)

    tmp = lookup(query, facts, nb_facts, kernel)
    print(tmp)
