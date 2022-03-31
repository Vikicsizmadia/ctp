# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ctp.clutrr.models.kb import BatchNeuralKB
from ctp.clutrr.models.util import uniform

from ctp.reformulators import BaseReformulator
from ctp.reformulators import GNTPReformulator

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class BatchHoppy(nn.Module):
    def __init__(self,
                 model: BatchNeuralKB,
                 hops_lst: List[Tuple[BaseReformulator, bool]], # [H,2], where H is the length of the hops list (--hops parameter); H BaseReformulator of given type (all are same, but reformulate goal to different number of subgoals)
                 k: int = 10, # when proving the body of a rule, we consider the k best substitutions for each variable
                 depth: int = 0, # how many times to reformulate the goal(s)
                 tnorm_name: str = 'min',
                 R: Optional[int] = None):
        super().__init__()

        self.model: BatchNeuralKB = model
        self.k = k

        self.depth = depth
        assert self.depth >= 0

        self.tnorm_name = tnorm_name
        assert self.tnorm_name in {'min', 'prod', 'mean'}

        self.R = R

        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])
        self.hops_lst = hops_lst

        logger.info(f'BatchHoppy(k={k}, depth={depth}, hops_lst={[h.__class__.__name__ for h in self._hops_lst]})')

    def _tnorm(self, x: Tensor, y: Tensor) -> Tensor:
        res = None
        # of each element of both tensor takes the smaller
        if self.tnorm_name == 'min':
            res = torch.min(x, y)
        elif self.tnorm_name == 'prod':
            res = x * y
        elif self.tnorm_name == 'mean':
            res = (x + y) / 2
        assert res is not None
        return res

    def prove(self,
              # rel: relation embeddings of the target subject,object paired with all possible relations - B relations
                    # --> [B,E]s concatenated
              # arg1,arg2: target subject,object embeddings (1-1 embedding) -> repeated B times
                    # --> all these [B,E] in a batch concatenated
              rel: Tensor, arg1: Tensor, arg2: Tensor, # [batch*B,E]; batch: number of batches; B: batch_size == number of relations
              # fact embeddings broke to 3 pieces: relation, arg1, arg2 embeddings
              # in 1 batch for all B instances the facts are the same
              # different batches possibly different facts
              facts: List[Tensor], # [story_rel, story_arg1, story_arg2]: [3,batch*B,F,E]
              # facts are padded, so each batch has the same number of facts
              # (in 1 batch there is originally the same facts for each B)
              # so nb_facts indicates the original number of facts
              nb_facts: Tensor, # [batch*B], the number of facts (before padding) in each instance
              # all entities corresponding to the facts of 1 batch --> this for all batches
              entity_embeddings: Tensor, # [batch*B,N,E], where each N is the same
              # entity_embeddings are padded (with 0-embeddings), so each batch has the same number of entities
              # (in 1 batch there is originally the same entities for each B)
              # so nb_entities indicates the original number of entities
              #  nb_entities: [batch*B], the number of entities (before padding) in each instance in each batch
              nb_entities: Tensor,
              depth:int) -> Tensor:
        # no reformulation
        scores_0 = self.model.score(rel, arg1, arg2, facts=facts, nb_facts=nb_facts,
                                    entity_embeddings=entity_embeddings, nb_entities=nb_entities)
        # reformulation
        scores_d = None
        if depth > 0:
            # 1 instead
            #scores_d = self.depth_r_score(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=depth)

###1s
            batch_size, embedding_size = rel.shape[0], rel.shape[1]
            entity_emb_max_nb = entity_embeddings.shape[1] # N
            global_res = None

            # [H,2]: list of reformulators (possibly reformulating into different dimensions, e.g. 2,2,1R)
            # and indicators if reversed (R)
            # enumerate H times: each is 1 Reformulator (hops_generator)
            for rule_idx, (hops_generator, is_reversed) in enumerate(self.hops_lst):
                sources, scores = arg1, None  # sources: [batch*B,E]

                # generates the new subgoals with the current reformulator
                # --> applies the transformator to each instance in the batch
                hop_rel_lst = hops_generator(rel)  # [nb_hops,batch*B,E]
                nb_hops = len(hop_rel_lst)  # usually: 2 (or 1)

                # enumerate through the newly generated subgoals
                # hop_rel: [batch*B,E] - 1st (then 2nd, 3rd,...) subgoal for each of the target relations in all the batches
                for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                    # [B * S, E] = [batch*B,E]
                    sources_2d = sources.view(-1, embedding_size)
                    print(f"sources_2d shape: {sources_2d.shape}")
                    nb_sources = sources_2d.shape[0]  # B*S = batch*B

                    nb_branches = nb_sources // batch_size  # 1, K, K*K, ...

                    # B = batch*B, S = K^n
                    # [B, E] --> [B, S, E] --> [B * S, N, E] --> [B * S * N, E]
                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:  # we are not at the last (batch of) subgoals

                        # DONE?
                        #TODO: entity embeddings for other argument that's None, so that we can call prove instead of r_forward

                        # [B, N, E] --> [B, S, N, E]
                        all_entities_3d = entity_embeddings.view(batch_size, 1, -1, embedding_size).repeat(1, nb_branches, 1, 1)
                        print(f"all_entities_3d shape: {all_entities_3d.shape}")
                        # [B * S * N, E]
                        all_entities_2d = all_entities_3d.view(-1, embedding_size)
                        print(f"all_entities_2d shape: {all_entities_2d.shape}")

                        # [B * S, E] --> [B * S, N, E] --> [B * S * N, E]
                        new_sources_3d = sources_2d.view(-1, 1, embedding_size).repeat(1, entity_emb_max_nb, 1)
                        print(f"new_sources_3d shape: {new_sources_3d.shape}")
                        new_sources_2d = new_sources_3d.view(-1, embedding_size)
                        print(f"new_sources_2d shape: {new_sources_2d.shape}")

                        hop_rel_3d = hop_rel_2d.view(-1, 1, embedding_size).repeat(1, entity_emb_max_nb, 1)
                        hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                        if is_reversed:
                            new_arg1 = all_entities_2d
                            new_arg2 = new_sources_2d
                        else:
                            new_arg1 = new_sources_2d
                            new_arg2 = all_entities_2d

                        # 2 instead
                        # [B * S, K], [B * S, K, E]
                        #z_scores, z_emb = self.r_hop(hop_rel_2d, new_arg1, new_arg2,
                        #                            facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

###2s

                        # [bathc*B, N]

                        # if depth == 0:
                        # returns the maximum similarity measure (=max proof score) of the queries (targets) to the facts
                        # --> does this for every entity embedding inserted in the place of 1 of the two arguments
                        # [batch*B,N]: entity embedding inserted in the place of the 2nd argument, [B,N]: -"- of the 1st argument

                        # if depth > 0:
                        # this gives the best scoring entity substitution's scores for each last entity embedding
                        # and from each of these the best among different Reformulators
                        # and from each of these the best among all depths<= "depth"
                        #print("r_hop calls r_forward with same depth: ", depth)
                        # 3 instead
                        #scores_sp, scores_po = self.r_forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings,
                        #                                      nb_entities, depth=depth)

###3s
                        # [batch*B,N]
                        # one of the arguments is all entity embeddings
                        new_scores = self.prove(hop_rel_2d, new_arg1, new_arg2, facts,
                                                nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        new_scores = new_scores.view(-1, entity_emb_max_nb)
###3f

                        #print("r_forward called by r_hop returned, was called with depth: ", depth)
                        #scores = scores_sp if arg2 is None else scores_po  # [batch*B,N]

                        print(f"scores.shape is: {new_scores.shape}")
                        k = min(self.k,
                                entity_emb_max_nb) # k (default 10), N (maximum number of entities in entity_embeddings)

                        # z_indices indicates which embedding substitution scored in the top k
                        # [batch*B, K], [batch*B, K]
                        # chooses the top k from each row in scores
                        z_scores, z_indices = torch.topk(new_scores, k=k, dim=1)

                        dim_1 = torch.arange(z_scores.shape[0], device=z_scores.device).view(-1, 1).repeat(1, k).view(-1)
                        dim_2 = z_indices.view(-1)

                        # making sure that we have enough entity embeddings by multiplicating them to match the size of z_scores
                        entity_embeddings, _ = uniform(z_scores, entity_embeddings)

                        # corresponding entity embeddings to the top k scores (z_scores)
                        z_emb = entity_embeddings[dim_1, dim_2].view(z_scores.shape[0], k, -1)  # [B,k,E]

                        # z_scores: [batch*B,K], z_emb: [batch*B,K,E]
###2f
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
                        # [B, S, E]
                        arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                        print(f"arg2_3d shape: {arg2_3d.shape}")
                        # [B * S, E]
                        arg2_2d = arg2_3d.view(-1, embedding_size)
                        print(f"arg2_2d shape: {arg2_2d.shape}")

                        # [B * S]
                        if is_reversed:
                            new_arg1, new_arg2 = arg2_2d, sources_2d
                        else:
                            new_arg1, new_arg2 = sources_2d, arg2_2d

                        # 4 instead
                        # z_scores_1d = self.r_score(hop_rel_2d, new_arg1, new_arg2,
                        #                             facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

###4s
                        print(f"hop_rel_2d shape: {hop_rel_2d.shape}")
                        z_scores_1d = self.prove(hop_rel_2d, new_arg1, new_arg2, facts,
                                                 nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
###4f

                        scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

                # finished enumerating through the new subgoals with current reformulator
                if scores is not None:
                    scores_2d = scores.view(batch_size, -1)
                    res, _ = torch.max(scores_2d, dim=1)
                else:
                    res = self.model.score(rel, arg1, arg2,
                                           facts=facts, nb_facts=nb_facts,
                                           entity_embeddings=entity_embeddings, nb_entities=nb_entities)

                global_res = res if global_res is None else torch.max(global_res, res)

            scores_d = global_res
###1f
        if scores_d is None:
            res = scores_0
        else:
            res = torch.max(scores_0, scores_d)  # choose the one with the higher score
        return res

    # depth_r_score calls this with depth-1
    # depth_r_forward calls this with depth-1
    def r_hop(self,
              # rel: [batch*B,E] - 1st (then 2nd, 3rd,...) subgoal for each of the target relations in all the batches
              # --> for the current Reformulator
              # arg1, arg2: [batch*B,E] --> target argument corresponding to the (now reformulated) rel
              # one of arg1,arg2 is None
              rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor], # [B,E]
              # [story_rel, story_arg1, story_arg2]: [3,batch*B,F,E], padded fact embeddings
              facts: List[Tensor], # [3,B,F,E] = [[B,F,E],[B,F,E],[B,F,E]], where F: maximum number of facts
              # (in 1 batch there is originally the same facts for each B)
              # [batch*B], the number of facts (before padding) in each instance
              nb_facts: Tensor, # [B], the number of facts in each instance of the batch
              # all entities corresponding to the facts of 1 batch --> this for all batches
              # [batch*B,N,E], where each N is the same
              entity_embeddings: Tensor, # [B,N,E], where N: maximum number of entities
              # (in 1 batch there is originally the same entities for each B)
              # [batch*B], the number of entities (before padding) in each instance in each batch
              nb_entities: Tensor, # [B], the number of entities in each instance of the batch
              depth: int) -> Tuple[Tensor, Tensor]:
        print("r_hop was called with depth: ", depth)
        assert (arg1 is None) ^ (arg2 is None)
        assert depth >= 0

        # batch_size: batch*B
        # embedding_size: E
        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # [bathc*B, N]

        # if depth == 0:
        # returns the maximum similarity measure (=max proof score) of the queries (targets) to the facts
        # --> does this for every entity embedding inserted in the place of 1 of the two arguments
        # [batch*B,N]: entity embedding inserted in the place of the 2nd argument, [B,N]: -"- of the 1st argument

        # if depth > 0:
        # this gives the best scoring entity substitution's scores for each last entity embedding
        # and from each of these the best among different Reformulators
        # and from each of these the best among all depths<= "depth"
        print("r_hop calls r_forward with same depth: ", depth)
        scores_sp, scores_po = self.r_forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=depth)
        print("r_forward called by r_hop returned, was called with depth: ", depth)
        scores = scores_sp if arg2 is None else scores_po # [batch*B,N]

        k = min(self.k, scores.shape[1]) # k (default 10), N (maximum number of entities in entity_embeddings)

# z_indices indicates which embedding substitution scored in the top k
        # [batch*B, K], [batch*B, K]
        # chooses the top k from each row in scores
        z_scores, z_indices = torch.topk(scores, k=k, dim=1)
        print("top k taken of scores calulated by r_forward")

        dim_1 = torch.arange(z_scores.shape[0], device=z_scores.device).view(-1, 1).repeat(1, k).view(-1)
        dim_2 = z_indices.view(-1)

        entity_embeddings, _ = uniform(z_scores, entity_embeddings) # making sure that we have enough entity embeddings by multiplicating them to match the size of z_scores

        # corresponding entity embeddings to the top k scores (z_scores)
        z_emb = entity_embeddings[dim_1, dim_2].view(z_scores.shape[0], k, -1) # [B,k,E]

        assert z_emb.shape[0] == batch_size
        assert z_emb.shape[2] == embedding_size

        print("final scores are getting returned from r_hop")
        return z_scores, z_emb # z_scores: [batch*B,K], z_emb: [batch*B,K,E]

    #called from clutrr-cli.py
    #
    def score(self,
              # rel: relation embeddings of the target subject,object paired with all possible relations - B relations
                    # --> [B,E]s concatenated
              # arg1,arg2: target subject,object embeddings (1-1 embedding) -> repeated B times
                    # --> all these [B,E] in a batch concatenated
              rel: Tensor, arg1: Tensor, arg2: Tensor, # [batch*B,E]; batch: number of batches; B: batch_size == number of relations
              # fact embeddings broke to 3 pieces: relation, arg1, arg2 embeddings
              # in 1 batch for all B instances the facts are the same
              # different batches possibly different facts
              facts: List[Tensor], # [story_rel, story_arg1, story_arg2]: [3,batch*B,F,E]
              # facts are padded, so each batch has the same number of facts
              # (in 1 batch there is originally the same facts for each B)
              # so nb_facts indicates the original number of facts
              nb_facts: Tensor, # [batch*B], the number of facts (before padding) in each instance
              # all entities corresponding to the facts of 1 batch --> this for all batches
              entity_embeddings: Tensor, # [batch*B,N,E], where each N is the same
              # entity_embeddings are padded (with 0-embeddings), so each batch has the same number of entities
              # (in 1 batch there is originally the same entities for each B)
              # so nb_entities indicates the original number of entities
              #  nb_entities: [batch*B], the number of entities (before padding) in each instance in each batch
              nb_entities: Tensor) -> Tensor:
        print('BatchHoppy.score called')
        print('BatchHoppy.score calls r_score with depth: ', self.depth)
        res = self.r_score(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=self.depth)
        print('BatchHoppy.score received r_score')
        return res

    def r_score(self,
                rel: Tensor, arg1: Tensor, arg2: Tensor,
                facts: List[Tensor],
                nb_facts: Tensor,
                entity_embeddings: Tensor,
                nb_entities: Tensor,
                depth: int) -> Tensor:
        res = None
        print('r_score calls depth_r_score with all depths smaller than or equal to ', depth)
        for d in range(depth + 1): # try all possible depths that are no deeper than "depth"
            print('depth_r_score called with depth: ', d)
            scores = self.depth_r_score(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=d)
            print('depth_r_score returned with depth: ', d)
            res = scores if res is None else torch.max(res, scores) # choose the one with the highest score
        return res

    def depth_r_score(self,
                      rel: Tensor, arg1: Tensor, arg2: Tensor, # [batch*B,E] --> target that we want to score
                      facts: List[Tensor], # [story_rel, story_arg1, story_arg2]: [3,batch*B,F,E]
                      nb_facts: Tensor, # [batch*B], the number of facts (before padding) in each instance
                      entity_embeddings: Tensor, # [batch*B,N,E]
                      nb_entities: Tensor, # [batch*B], the number of entities (before padding) in each instance
                      depth: int) -> Tensor: # result is [B]
        assert depth >= 0

        if depth == 0:
            print('depth_r_score with depth 0 calls model.score (BatchNeuralKB)')
            return self.model.score(rel, arg1, arg2,
                                    facts=facts, nb_facts=nb_facts,
                                    entity_embeddings=entity_embeddings, nb_entities=nb_entities)

        # batch_size: batch*B
        # embedding_size: E
        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        global_res = None

        mask = None

        # [H,2]: list of reformulators (possibly reformulating into different dimensions, e.g. 2,2,1R)
        # and indicators if reversed (R)
        new_hops_lst = self.hops_lst

        # enumerate H times: each is 1 Reformulator (hops_generator)
        print('depth_r_score enumerates through new_hops_lst: ', new_hops_lst)
        for rule_idx, (hops_generator, is_reversed) in enumerate(new_hops_lst):
            sources, scores = arg1, None # sources: [batch*B,E]
            print("depth_r_score current hops_generator: ", hops_generator)

            # generates the new subgoals with the current reformulator
            # --> applies the transformator to each instance in the batch
            hop_rel_lst = hops_generator(rel) # [nb_hops,batch*B,E]
            nb_hops = len(hop_rel_lst) # usually: 2 (or 1)
            print("new subgoals generated, their number is: ", nb_hops)

            # enumerate through the newly generated subgoals
            # hop_rel: [batch*B,E] - 1st (then 2nd, 3rd,...) subgoal for each of the target relations in all the batches
            print('enumerating through the new subgoals')
            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                print('current subgoal number is: ', hop_idx)
                # [B * S, E] = [batch*B,E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0] # B*S = batch*B


                nb_branches = nb_sources // batch_size # 1, K, K*K, ...
                print('number of branches: ', nb_branches)

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1) # [batch*B,1,E]
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size) # [batch*B,E], same as hop_rel

                if hop_idx < nb_hops: # we are not at the last (batch of) subgoals
                    # [B * S, K], [B * S, K, E]
                    print("we are not at the last subgoal, so depth_r_score calls r_hop with depth-1: ", depth-1)
                    if is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                     facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                     facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    print("depth_r_score received the result from r_hop called with depth: ", depth-1)
                    k = z_emb.shape[1]

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
                    # [B, S, E]
                    arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)

                    # [B * S]
                    print("we are at the last subgoal, so depth_r_score calls r_score with depth-1: ", depth - 1)
                    if is_reversed:
                        z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d,
                                                   facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    else:
                        z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d,
                                                   facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    print("depth_r_score received the result from r_score called with depth: ", depth - 1)
                    scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

            print("finished enumerating through the new subgoals with current reformulator")
            if scores is not None:
                scores_2d = scores.view(batch_size, -1)
                res, _ = torch.max(scores_2d, dim=1)
            else:
                print("scores was None, so model.score gets called")
                res = self.model.score(rel, arg1, arg2,
                                       facts=facts, nb_facts=nb_facts,
                                       entity_embeddings=entity_embeddings, nb_entities=nb_entities)
                print("model.score returned")

            global_res = res if global_res is None else torch.max(global_res, res)
            print("updated global result score")

        print("final depth_r_score result calculated")
        return global_res

    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                facts: List[Tensor],
                nb_facts: Tensor,
                entity_embeddings: Tensor,
                nb_entities: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        print("forward called")
        print("forward calls r_forward with depth: ", self.depth)
        res_sp, res_po = self.r_forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=self.depth)
        print("r_forward returned to forward called with depth: ", self.depth)
        print("result scores are getting returned by forward")
        return res_sp, res_po

    # called by r_hop with same depth (depth_r_score depth-1)
    def r_forward(self,
                  # rel: [batch*B,E] - 1st (then 2nd, 3rd,...) subgoal for each of the target relations in all the batches
                  # --> for the current Reformulator
                  # arg1, arg2: [batch*B,E] --> target argument corresponding to the (now reformulated) rel
                  # one of arg1,arg2 is None
                  rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],  # [B,E]
                  # [story_rel, story_arg1, story_arg2]: [3,batch*B,F,E], padded fact embeddings
                  facts: List[Tensor],  # [3,B,F,E] = [[B,F,E],[B,F,E],[B,F,E]], where F: maximum number of facts
                  # (in 1 batch there is originally the same facts for each B)
                  # [batch*B], the number of facts (before padding) in each instance
                  nb_facts: Tensor,  # [B], the number of facts in each instance of the batch
                  # all entities corresponding to the facts of 1 batch --> this for all batches
                  # [batch*B,N,E], where each N is the same
                  entity_embeddings: Tensor,  # [B,N,E], where N: maximum number of entities
                  # (in 1 batch there is originally the same entities for each B)
                  # [batch*B], the number of entities (before padding) in each instance in each batch
                  nb_entities: Tensor,  # [B], the number of entities in each instance of the batch
                  # same depth as r_hop
                  depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        print("r_forward was called with depth: ", depth)
        res_sp, res_po = None, None
        print("r_forward iterates through depths no larger than given depth, each time calls depth_r_forward")
        for d in range(depth + 1): # iterate through all possible depths no larger than "depth"
            # [B,N], [B,N]
            # this gives the best scoring entity substitution's scores for each last entity embedding
            # and from each of these the best among different Reformulators
            print("r_forward calls depth_r_forward with depth: ", d)
            scores_sp, scores_po = self.depth_r_forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=d)
            print("depth_r_forward returned to r_forward, was called with depth: ", d)
            res_sp = scores_sp if res_sp is None else torch.max(res_sp, scores_sp) # gives back from each element the max of the 2 tensors
            res_po = scores_po if res_po is None else torch.max(res_po, scores_po)
            print("received scores are updated with torch.max")
        # [B,N], [B,N]
        # this gives the best scoring entity substitution's scores for each last entity embedding
        # and from each of these the best among different Reformulators
        # and from each of these the best among all depths
        print("scores are getting returned from r_forward that was called with depth: ", depth)
        return res_sp, res_po


    # called by r_forward, same parameters, just depth iterates in range(depth+1)

    # returns the maximum similarity measure (=max proof score) of the queries (targets) to the facts
    # --> does this for every entity embedding inserted in the place of 1 of the two arguments
    # [B,N]: entity embedding inserted in the place of the 2nd argument, [B,N]: -"- of the 1st argument
    def depth_r_forward(self,
                        rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                        facts: List[Tensor],
                        nb_facts: Tensor,
                        entity_embeddings: Tensor,
                        nb_entities: Tensor,
                        depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]: #result: [B,N],[B,N]

        print("depth_r_forward called with depth: ", depth)
        # batch_size: batch*B
        # embedding_size: E
        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # returns the maximum similarity measure (=max proof score) of the queries (targets) to the facts
        # --> does this for every entity embedding inserted in the place of 1 of the two arguments
        # [B,N]: entity embedding inserted in the place of the 2nd argument, [B,N]: -"- of the 1st argument
        if depth == 0:
            print("depth_r_forward with depth 0 calls model.forward (BatchNeuralKB) and returns the received scores")
            return self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities) #score_sp, score_po

        global_scores_sp = global_scores_po = None

        mask = None
        # [H,2]: list of reformulators (possibly reformulating into different dimensions, e.g. 2,2,1R)
        # and indicators if reversed (R)
        new_hops_lst = self.hops_lst

# IGNORE
#        if self.R is not None:
#            batch_rules_scores = torch.cat([h.prior(rel).view(-1, 1) for h, _ in self.hops_lst], 1)
#            topk, indices = torch.topk(batch_rules_scores, self.R)

            # [R x E]
#            rule_heads = torch.cat([h.head for h, _ in self.hops_lst], dim=0)
#            rule_body1s = torch.cat([h.memory_lst[0] for h, _ in self.hops_lst], dim=0)
#            rule_body2s = torch.cat([h.memory_lst[1] for h, _ in self.hops_lst], dim=0)

#            kernel = self.hops_lst[0][0].kernel
            # the top k embeddings from the generated heads that scored the highest when compared to rel
            # (separately for each instance in the batch)
#            new_rule_heads = F.embedding(indices, rule_heads) # [B,R,E]
#            new_rule_body1s = F.embedding(indices, rule_body1s) # [B,R,E]
#            new_rule_body2s = F.embedding(indices, rule_body2s) # [B,R,E]

#            assert new_rule_heads.shape[1] == self.R

#            new_hops_lst = []
#            for i in range(new_rule_heads.shape[1]):
#                r = GNTPReformulator(kernel=kernel, head=new_rule_heads[:, i, :],
#                                     body=[new_rule_body1s[:, i, :], new_rule_body2s[:, i, :]])
#                new_hops_lst += [(r, False)]

        # runs H times through H BaseReformulators (any subclass) --> hop_generators is 1 BaseReformulator
        # rule_idx is the number of which BR we are currently at
        print("depth_r_forward enumerates through new_hops_lst: ", new_hops_lst)
        for rule_idx, (hop_generators, is_reversed) in enumerate(new_hops_lst):
            print("depth_r_forward current hop_generator: ", hop_generators)
            scores_sp = scores_po = None
            # rel: [batch * B, E]
            # generates the new subgoals with the current reformulator
            # --> applies the transformator to each instance in the batch
            hop_rel_lst = hop_generators(rel) # [nb_hops,batch*B,E], only need rel for the batch dimension in some cases
            nb_hops = len(hop_rel_lst)
            print("new subgoals generated, their number is: ", nb_hops)


            # either arg1 or arg2 will be None!
            if arg1 is not None:
                print("arg1 is not None")
                # new:
                # sources: [branch*B,E]
                sources, scores = arg1, None # sources: [B,branches,E], where branches (=S) are the number of branches in an instance

                # enumerate through the newly generated subgoals
                # hop_rel: [batch*B,E] - 1st (then 2nd, 3rd,...) subgoal for each of the target relations in all the batches
                print("enumerating through the new subgoals")
                for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1): # iterates nb_hops times (which is usually 2)
                    print("current subgoal number is: ", hop_idx)
                    # [B * S, E], where S is the number of branches
                    # new: [branch*B,E] then later [batch * B * K, E]
                    sources_2d = sources.view(-1, embedding_size) # flat out 3d sources
                    # new: branch*B then later batch*B*K
                    nb_sources = sources_2d.shape[0] # = batch size (B) * branches

                    # new: 1 then later K
                    nb_branches = nb_sources // batch_size # S

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1) # [batch*B,1,E] then later [batch*B,K,E]
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size) # [batch*B,E], same as hop_rel; then later [batch*B*K,E]

                    if hop_idx < nb_hops: # if we are not at the last subgoal
                        # [B * S, K], [B * S, K, E]
                        # z_scores: [batch*B,K], z_emb: [batch*B,K,E]
                        print("we are not at the last subgoal, so depth_r_forward calls r_hop with depth-1: ", depth - 1)
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        print("depth_r_forward received the result from r_hop called with depth: ", depth - 1)
                        k = z_emb.shape[1] # k

                        # [B * S * K]
                        # [batch * B * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        # [batch * B * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        # [batch * B * K, E]
# here is the indication of which embedding substitutions were used
                        sources = z_emb_2d
                        # [B * S * K]
                        # [batch * B * K], then [batch * B * K^2], ...
                        # takes the minimum (if that's tnorm) of the embedding substitution scores
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                        print("took the min of new scores and the previous subgoals' minimum score (with _tnorm)")
                    else:
                        # [B * S, N]
                        # scores_sp: [batch*B,N]
                        print("we are at the last subgoal, so depth_r_forward calls r_forward with depth-1: ", depth - 1)
                        if is_reversed:
                            _, scores_sp = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            scores_sp, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        print("depth_r_forward received the result from r_forward called with depth: ", depth - 1)
                        # difference between r_hop and r_forward is that r_hop takes only the top k scores,
                        # while r_forward gives back all scores

                        nb_entities_ = scores_sp.shape[1] # N

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities_)
                            # now comparing entity substitution top k scores with all possible entity subst. scores
                            scores_sp = self._tnorm(scores, scores_sp) # (usually) taking the min of them (minimum of all embedding similarities)
                            print("took the min of new scores and the previous subgoal's minimum score (with _tnorm)")
                            # [B, S, N], where S = K^(nb_hops-1)
                            scores_sp = scores_sp.view(batch_size, -1, nb_entities_)
                            # [B, N]
                            # this gives the best scoring entity substitution's scores for each last entity embedding
                            scores_sp, _ = torch.max(scores_sp, dim=1) # taking only the max from the branches (maximum of all proof scores)s
                            print("took the max scores of the branches in the batches: scores_sp")

            if arg2 is not None:
                print("arg2 is not None")
                sources, scores = arg2, None

# TO DELETE
#                prior = hop_generators.prior(rel)
#                if prior is not None:
#                    scores = prior
                # scores = hop_generators.prior(rel)

                print("enumerating through the new subgoals but starting from the last one")
                for hop_idx, hop_rel in enumerate(reversed([h for h in hop_rel_lst]), start=1):
                    print("current subgoal number is: ", hop_idx)
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        print("we are not at the last subgoal, so depth_r_forward calls r_hop with depth-1: ", depth - 1)
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        print("depth_r_forward received the result from r_hop called with depth: ", depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                        print("took the min of new scores and the previous subgoals' minimum score (with _tnorm)")
                    else:
                        # [B * S, N]
                        print("we are at the last subgoal, so depth_r_forward calls r_forward with depth-1: ", depth - 1)
                        if is_reversed:
                            scores_po, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            _, scores_po = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        print("depth_r_forward received the result from r_forward called with depth: ", depth - 1)
                        nb_entities_ = scores_po.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities_)
                            scores_po = self._tnorm(scores, scores_po)
                            print("took the min of new scores and the previous subgoal's minimum score (with _tnorm)")
                            # [B, S, N]
                            scores_po = scores_po.view(batch_size, -1, nb_entities_)
                            # [B, N]
                            # this gives the best scoring entity substitution's scores for each last entity embedding
                            scores_po, _ = torch.max(scores_po, dim=1)
                            print("took the max scores of the branches in the batches: scores_po")

            if scores_sp is None and scores_po is None:
                print("both scores_sp and scores_po is None")
                print("depth_r_forward calls model.forward (BatchNeuralKB)")
                scores_sp, scores_po = self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)
                print("depth_r_forward receives result from model.forward")

            # here we have the scores for running 1 Reformulator at the current depth
            # --> we have to compare this score with previous Reformulator score --> keep the best for each
            # [B,N]
            # this gives the best scoring entity substitution's scores for each last entity embedding
            # and from each of these the best among different Reformulators
            print("depth_r_forward at the end of 1 enumeration through new_hops_lst takes the max global score so far (torch.max)")
            global_scores_sp = scores_sp if global_scores_sp is None else torch.max(global_scores_sp, scores_sp)
            global_scores_po = scores_po if global_scores_po is None else torch.max(global_scores_po, scores_po)

        if global_scores_sp is None and global_scores_po is None:
            print("both global_scores_sp and global_scores_po are None, so depth_r_forward calls model.forward (BatchNeuralKB)")
            global_scores_sp, global_scores_po = self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)
            print("depth_r_forward receives result scores from model.forward")

        print("depth_r_forward returns final scores")
        return global_scores_sp, global_scores_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        print("factor function was called")
        return self.model.factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor]) -> List[Tensor]:
        print("extra_factors function was called")
        return [hop_generator(rel) for hop_generators in self.hops_lst for hop_generator in hop_generators]
