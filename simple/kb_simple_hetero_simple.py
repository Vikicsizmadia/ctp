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

        # TODO: figure out if scoring_type needed
        """Initializing some parameters of the BatchNeuralKB instance.

            Args:
                kernel (BaseKernel): The kernel used for calculating the scores of the queries.
                scoring_type (str):
                    Default: 'concat'.
        """

        super().__init__()

        self.kernel = kernel
        self.scoring_type = scoring_type
        assert self.scoring_type in {'concat'}

    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,
              facts: List[Tensor]) -> Tensor:

        """For the batch of queries calculates the batch of the corresponding maximum similarity measures
        (=max proof scores) with the facts.

            Args:
                rel (Tensor): rel, arg1, arg2 are the relation, subject, object embeddings.
                    These are the relations of the query embeddings, each relation corresponding
                        to the subject, object embeddings at the same place in the Tensors,
                        making up the query "rel(arg1,arg2)".
                    Shape: [B,E] - B embeddings, E embedding size.
                arg1 (Tensor): Target subject embeddings corresponding to the relations in rel.
                    Shape: [B,E]
                arg2 (Tensor): Target object embeddings corresponding to the relations in rel.
                    Shape: [B,E]
                facts (List[Tensor]): [fact_rel, fact_arg1, fact_arg2].
                    Fact embeddings broke to 3 pieces: relation, arg1, arg2 embeddings.
                    These same facts are common to all queries in the batch.
                    Shape: [3,F,E] - F is the number of facts.

            Returns:
                Tensor of the maximum similarity measure (=max proof score) of the queries (rel,arg1,arg2) to the facts.
                Shape: [B] - scores for each query.
        """

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
        # TODO: figure out if factor needed
        return embedding_vector
