# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor
from kb_final import BatchNeuralKB
from ctp.clutrr.models.util import uniform

from ctp.reformulators import BaseReformulator

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class BatchHoppy(nn.Module):
    """The class responsible for the proof score calculation part of the CTP method."""

    def __init__(self,
                 model: BatchNeuralKB,
                 hops_lst: List[Tuple[BaseReformulator, bool]],
                 k: int = 10,
                 depth: int = 0,
                 tnorm_name: str = 'min'):

        """Initializing some parameters of the BatchHoppy instance.

            Args:
                model (BatchNeuralKB): The knowledge base used for scoring the queries.
                hops_lst (List[Tuple[BaseReformulator, bool]]): List of BaseReformulators and indicators if reversed.
                    All BaseReformulators are of the same type given by the reformulator_name argument in main.
                    Each BaseReformulator reformulates goals into different number of subgoals given by the
                        corresponding element in the hops_str argument in main.
                    If bool is True then the reformulated subgoal arguments are reversed.
                    Shape: [H,2] - H BaseReformulators (length of hops_str) and indicators if reversed.
                k (int): When proving the body of a rule, we will consider the k best substitutions for each variable.
                    Default: 10.
                depth (int): The total number of how many times to reformulate the goal(s).
                    When the reformulation takes place this will be the maximum depth, and each individual subgoal
                        will be reformulated to the depth giving the highest score.
                    Default: 0.
                tnorm_name (str): This gives the method of how to update scores while reformulating the current
                    goal into subgoals.
                    'min': The function updates scores by taking the minimum of all embedding similarity scores so far.
                    'prod': The function updates scores by taking the product of the embedding similarity scores so far.
                    'mean': The function updates scores by taking the mean of all embedding similarity scores so far.
                    Default: 'min'.
        """

        super().__init__()

        self.model: BatchNeuralKB = model
        self.k = k
        self.depth = depth
        assert self.depth >= 0
        self.tnorm_name = tnorm_name
        assert self.tnorm_name in {'min', 'prod', 'mean'}
        self.hops_lst = hops_lst

        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])
        logger.info(f'BatchHoppy(k={k}, depth={depth}, hops_lst={[h.__class__.__name__ for h in self._hops_lst]})')

    def _tnorm(self, x: Tensor, y: Tensor) -> Tensor:
        """Updates previous embedding similarity scores with newly calculated similarity scores with the given method.

            Args:
                x (Tensor): Tensor of proof scores so far. Must be same shape as y.
                y (Tensor): Tensor of proof scores so far. Must be same shape as x.

            Returns:
                Tensor of the updated proof scores of the same shape as the input Tensors.
        """

        res = None
        # takes the smaller of each element of both tensor
        if self.tnorm_name == 'min':
            res = torch.min(x, y)
        elif self.tnorm_name == 'prod':
            res = x * y
        elif self.tnorm_name == 'mean':
            res = (x + y) / 2
        assert res is not None
        return res

    def prove(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,
              facts: List[Tensor],
              nb_facts: Tensor,
              entity_embeddings: Tensor,
              nb_entities: Tensor,
              depth: int) -> Tensor:

        """Does the main part of the CTP method by calculating the proof scores of the given queries.

            Args:
                rel (Tensor): rel, arg1, arg2 are the relation, subject, object embeddings.
                    These are the relations of the query embeddings, each relation corresponding
                        to the subject, object embeddings at the same place in the Tensors,
                        making up the query "rel(arg1,arg2)".
                    In the first call of prove every query subject and object in each batch ('batch' number
                        of batches) are paired with every possible relation type ('R' different types) to give
                        the query Tensor the size [batch*R,E] = [B,E].
                    Shape: [B,E] - B embeddings, E embedding size.
                arg1 (Tensor): Target subject embeddings corresponding to the relations in rel.
                    Shape: [B,E]
                arg2 (Tensor): Target object embeddings corresponding to the relations in rel.
                    Shape: [B,E]
                facts (List[Tensor]): [fact_rel, fact_arg1, fact_arg2].
                    Fact embeddings broke to 3 pieces: relation, arg1, arg2 embeddings.
                    Note: In 1 batch for all R instances the facts are the same.
                        In different batches there are possibly different facts.
                    Shape: [3,batch*R,F,E] = [3,B,F,E] - F is the maximum number of facts among all batches.
                nb_facts (Tensor): The original number of facts (before padding) for each batch.
                    'facts' are padded (with 0-embeddings), so each batch has the same number of facts
                        for later convenience.
                    Shape: [batch*R] = [B]
                entity_embeddings (Tensor): Entity embeddings corresponding to the facts of each batch.
                    Shape: [batch*R,N,E] = [B,N,E] - N is the maximum number of entity embeddings among all batches.
                nb_entities (Tensor): The original number of entities (before padding) for each batch.
                    'entity_embeddings' are padded (with 0-embeddings), so each batch has the same
                        number of entities for later convenience.
                    Shape: [batch*R] = [B]
                depth (int): The number of how many times to reformulate the goal(s).
                    When the reformulation takes place this will be the maximum depth, and each individual subgoal
                        will be reformulated to the depth giving the highest score.
                    This will decrease by 1 at every recursive call.

            Returns:
                Tensor of the maximum similarity measure (=max proof score) of the queries (rel,arg1,arg2) to the facts.
                Shape: [B] - scores for each query.
        """

        # no reformulation

        # [B]
        scores_0 = self.model.score(rel, arg1, arg2, facts=facts, nb_facts=nb_facts)

        # reformulation

        scores_d = None
        if depth > 0:

            # batch_size: B, embedding_size: E, entity_emb_max_nb: N
            batch_size, embedding_size = rel.shape[0], rel.shape[1]
            entity_emb_max_nb = entity_embeddings.shape[1]

            # need to have the entity_embeddings of the same batch size as rel
            entity_embeddings, _ = uniform(rel, entity_embeddings)

            global_res = None

            # enumerate H times: each is 1 Reformulator (hops_generator)
            for rule_idx, (hops_generator, is_reversed) in enumerate(self.hops_lst):
                sources, scores = arg1, None  # sources: [B,E]

                # generates the new subgoals with the current reformulator
                # --> applies the transformator to each instance in the batches
                hop_rel_lst = hops_generator(rel)  # [nb_hops,B,E]
                nb_hops = len(hop_rel_lst)  # usually: 2 (or 1)

                # enumerate through the newly generated subgoals
                # hop_rel: [B,E] - 1st (then 2nd, 3rd,...) subgoal for each of the target relations in all the batches
                for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                    # [B * S, E], where S: 1, K^1, K^2, ... in the consecutive iterations
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]  # B*S

                    nb_branches = nb_sources // batch_size  # called S: 1, K^1, K^2, ...

                    # [B, E] --> [B, S, E] --> [B * S, N, E] --> [B * S * N, E]
                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:  # we are not at the last (batch of) subgoals
                        # [B, N, E] --> [B, S, N, E] --> [B * S * N, E]
                        all_entities_3d = entity_embeddings.view(batch_size, 1, -1, embedding_size).repeat(1, nb_branches, 1, 1)
                        all_entities_2d = all_entities_3d.view(-1, embedding_size)

                        # [B * S, E] --> [B * S, N, E] --> [B * S * N, E]
                        new_sources_3d = sources_2d.view(-1, 1, embedding_size).repeat(1, entity_emb_max_nb, 1)
                        new_sources_2d = new_sources_3d.view(-1, embedding_size)

                        # [B * S, E] --> [B * S, N, E] --> [B * S * N, E]
                        hop_rel_3d = hop_rel_2d.view(-1, 1, embedding_size).repeat(1, entity_emb_max_nb, 1)
                        hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                        if is_reversed:
                            new_arg1, new_arg2 = all_entities_2d, new_sources_2d
                        else:
                            new_arg1, new_arg2 = new_sources_2d, all_entities_2d

                        # one of the arguments is all entity embeddings
                        # [B * S, N]
                        new_scores = self.prove(hop_rel_2d, new_arg1, new_arg2, facts,
                                                nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        new_scores = new_scores.view(-1, entity_emb_max_nb)

                        # k (default 10), N (maximum number of entities in entity_embeddings)
                        k = min(self.k, entity_emb_max_nb)

                        # z_indices indicates which embedding substitution scored in the top k
                        # chooses the top k from each row in new_scores
                        # [B * S, K], [B * S, K]
                        z_scores, z_indices = torch.topk(new_scores, k=k, dim=1)

                        dim_1 = torch.arange(z_scores.shape[0], device=z_scores.device).view(-1, 1).repeat(1, k).view(-1)
                        dim_2 = z_indices.view(-1)

                        # making sure that we have enough entity embeddings by multiplicating them
                        # to match the size of z_scores
                        entity_embeddings, _ = uniform(z_scores, entity_embeddings)

                        # corresponding entity embeddings to the top k scores (z_scores)
                        # [B * S, K, E]
                        z_emb = entity_embeddings[dim_1, dim_2].view(z_scores.shape[0], k, -1)

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B, E] --> [B, S, E] --> [B * S, E]
                        arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                        arg2_2d = arg2_3d.view(-1, embedding_size)

                        # [B * S]
                        if is_reversed:
                            new_arg1, new_arg2 = arg2_2d, sources_2d
                        else:
                            new_arg1, new_arg2 = sources_2d, arg2_2d

                        # one of the arguments is the arg2 entities from the query
                        # [B * S]
                        z_scores_1d = self.prove(hop_rel_2d, new_arg1, new_arg2, facts,
                                                 nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

                        # [B * S]
                        scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

                # finished enumerating through the new subgoals with current reformulator

                # take maximum scores from all branches
                # [B * S] --> [B, S] --> [B]
                if scores is not None:
                    scores_2d = scores.view(batch_size, -1)
                    res, _ = torch.max(scores_2d, dim=1)
                else:
                    res = self.model.score(rel, arg1, arg2, facts=facts, nb_facts=nb_facts)

                # update scores with scores obtained from using the current reformulator
                # [B]
                global_res = res if global_res is None else torch.max(global_res, res)

            # [B]
            scores_d = global_res

        # [B]
        if scores_d is None:
            res = scores_0
        else:
            res = torch.max(scores_0, scores_d)  # choose the one with the higher score
        return res

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor]) -> List[Tensor]:
        return [hop_generator(rel) for hop_generators in self.hops_lst for hop_generator in hop_generators]
