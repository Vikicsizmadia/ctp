# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.kernels import BaseKernel
from ctp.clutrr.models.util import lookup, uniform

from typing import List, Tuple, Optional

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
              nb_facts: Tensor, # [B]
              entity_embeddings: Optional[Tensor] = None,
              nb_entities: Optional[Tensor] = None) -> Tensor:

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

    # returns the maximum similarity measure (=max proof score) of the queries to the facts
    # --> does this for every entity embedding inserted in the place of 1 of the two arguments
    # [B,N]: entity embedding inserted in the place of the 2nd argument, [B,N]: -"- of the 1st argument
    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor], # [B,E]
                facts: List[Tensor], # [3,B,F,E], F: maximum number of facts
                nb_facts: Tensor, # [B]
                entity_embeddings: Tensor, # [B,N,E], N: maximum number of entities
                nb_entities: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]: # [B]
        # rel: [B, E], arg1: [B, E], arg2: [B, E]
        # facts: [B, F, E]
        # entity_embeddings: [B, N, E] (XXX: need no. entities)

        # [B, F, 3E]
        fact_emb = torch.cat(facts, dim=2)

        fact_emb, nb_facts = uniform(rel, fact_emb, nb_facts)
        entity_embeddings, nb_entities = uniform(rel, entity_embeddings, nb_entities)

        batch_size = rel.shape[0] # B
        embedding_size = rel.shape[1] # E
        entity_size = entity_embeddings.shape[1] # N
        fact_size = fact_emb.shape[1] # F

        # [B, N, F, 3E]
        fact_bnf3e = fact_emb.view(batch_size, 1, fact_size, -1).repeat(1, entity_size, 1, 1)

        # [B, N, F, E]
        rel_bnfe = rel.view(batch_size, 1, 1, embedding_size).repeat(1, entity_size, fact_size, 1)

        # [B, N, F, E]
        emb_bnfe = entity_embeddings.view(batch_size, entity_size, 1, embedding_size).repeat(1, 1, fact_size, 1)

        # [B, F]
        fact_mask = torch.arange(fact_size, device=nb_facts.device).expand(batch_size, fact_size) < nb_facts.unsqueeze(1)
        # [B, N]
        entity_mask = torch.arange(entity_size, device=nb_entities.device).expand(batch_size, entity_size) < nb_entities.unsqueeze(1)

        # [B, N, F]
        mask = fact_mask.view(batch_size, 1, fact_size).repeat(1, entity_size, 1) * \
               entity_mask.view(batch_size, entity_size, 1).repeat(1, 1, fact_size)

        score_sp = score_po = None # subject score (sp) = first arg, object score (po) = second arg

        if arg1 is not None:
            # [B, N, F, E]
            arg1_bnfe = arg1.view(batch_size, 1, 1, embedding_size).repeat(1, entity_size, fact_size, 1)

            # [B, N, F, 3E]
            query_bnf3e = torch.cat([rel_bnfe, arg1_bnfe, emb_bnfe], dim=3)

            # [B, N, F]
            scores_bnf = self.kernel(query_bnf3e, fact_bnf3e).view(batch_size, entity_size, fact_size)
            scores_bnf = scores_bnf * mask

            # [B, N]
            score_sp, _ = torch.max(scores_bnf, dim=2)

        if arg2 is not None:
            # [B, N, F, E]
            arg2_bnfe = arg2.view(batch_size, 1, 1, embedding_size).repeat(1, entity_size, fact_size, 1)

            # [B, N, F, 3E]
            query_bnf3e = torch.cat([rel_bnfe, emb_bnfe, arg2_bnfe], dim=3)

            # [B, N, F]
            scores_bnf = self.kernel(query_bnf3e, fact_bnf3e).view(batch_size, entity_size, fact_size)
            scores_bnf = scores_bnf * mask

            # [B, N]
            score_po, _ = torch.max(scores_bnf, dim=2)

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return embedding_vector
