# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.kernels import BaseKernel
from ctp.clutrr.models.util import lookup

from typing import List, Optional

import logging

logger = logging.getLogger(__name__)


class BatchNeuralKB(nn.Module):
    def __init__(self,
                 kernel: BaseKernel,
                 scoring_type: str = 'concat'):
        super().__init__()

        self.kernel = kernel
        self.scoring_type = scoring_type
        assert self.scoring_type in {'concat'}

    # gives back for the batch of queries the batch of the corresponding maximum similarity measures (=max proof scores) with the facts
    # [B]
    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor, # [B,E]
              facts: List[Tensor], # [3,B,F,E]
              nb_facts: Tensor) -> Tensor: # [B]

        # [B, F, 3E]
        facts_emb = torch.cat(facts, dim=2)

        # [B, 3E]
        batch_emb = torch.cat([rel, arg1, arg2], dim=1)

        # [B, F]
        # the similarity measures of each fact in facts to the corresponding query rule in the batch
        batch_fact_scores = lookup(batch_emb, facts_emb, nb_facts, self.kernel)

        # [B]
        res, _ = torch.max(batch_fact_scores, dim=1)
        return res

    def factor(self, embedding_vector: Tensor) -> Tensor:
        return embedding_vector
