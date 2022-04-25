# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.kernels import BaseKernel

from typing import List

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
              rel: Tensor, arg1: Tensor, arg2: Tensor,  # [B,E]
              facts: List[Tensor] ) -> Tensor:  # [3,F,E]

        # [F, 3E]
        facts = torch.cat(facts, dim=1)

        # [B, 3E]
        query = torch.cat([rel, arg1, arg2], dim=1)

        # [B, F]
        # the similarity measures of each fact in facts to the corresponding query rule in the batch
        batch_size, fact_size = query.shape[0], facts.shape[0]
        batch_fact_scores = self.kernel(query, facts).view(batch_size, fact_size)

        # [B]
        res, _ = torch.max(batch_fact_scores, dim=1)
        return res

    def factor(self, embedding_vector: Tensor) -> Tensor:
        return embedding_vector
