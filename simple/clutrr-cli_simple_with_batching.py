#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join, dirname, abspath
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F

from ctp.util import make_batches
from ctp.clutrr import Fact, Data, Instance
from simple import DataParser, accuracy

from ctp.clutrr.models import BatchNeuralKB
from model_simple_cleaned import BatchHoppy

from ctp.reformulators import BaseReformulator
from ctp.reformulators import StaticReformulator
from ctp.reformulators import LinearReformulator
from ctp.reformulators import AttentiveReformulator
from ctp.reformulators import MemoryReformulator
from ctp.reformulators import NTPReformulator

from ctp.kernels import BaseKernel, GaussianKernel
from ctp.regularizers import N2, N3, Entropy

from typing import List, Tuple, Dict, Optional

from torch_geometric.data import HeteroData

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=256, precision=4, suppress=True, threshold=sys.maxsize)

torch.set_num_threads(multiprocessing.cpu_count())


def decode(vector: Tensor,
           kernel: BaseKernel,
           relation_embeddings: nn.Module) -> Tuple[int, float]:
    weight = relation_embeddings.weight
    k = kernel.pairwise(vector, weight)[0, :]
    top_idx = k.argmax(dim=0).item()
    top_score = k[top_idx].item()
    return top_idx, top_score


def show_rules(model: BatchHoppy,
               kernel: BaseKernel,
               relation_embeddings: nn.Embedding,
               relation_to_idx: Dict[str, int],
               device: Optional[torch.device] = None):
    idx_to_relation = {i: r for r, i in relation_to_idx.items()}

    rel_idx_pair_lst = sorted(relation_to_idx.items(), key=lambda kv: kv[1])

    for r, i in rel_idx_pair_lst:
        indices = torch.tensor([i], dtype=torch.long, device=device)

        r_emb = relation_embeddings(indices)

        hops_lst = [p for p in model.hops_lst]

        for reformulator, _ in hops_lst:
            def _to_pair(hop: Tensor) -> Tuple[str, float]:
                idx, score = decode(hop, kernel, relation_embeddings)
                rel = idx_to_relation[idx]
                return rel, score

            hop_tensor_lst = [hop for hop in reformulator(r_emb)]

            r_hops = [_to_pair(hop) for hop in hop_tensor_lst]
            print(r, ' ← ', ', '.join(f'({a} {b:.4f})' for a, b in r_hops))

            if isinstance(model.model, BatchHoppy):
                for _r_emb in hop_tensor_lst:
                    _hops_lst = [p for p in model.model.hops_lst]

                    for j, (_reformulator, _) in enumerate(_hops_lst):
                        _hop_tensor_lst = [_hop for _hop in _reformulator(_r_emb)]
                        _r_hops = [_to_pair(_hop) for _hop in _hop_tensor_lst]
                        print(j, ' ← ', ', '.join(f'({_a} {_b:.4f})' for _a, _b in _r_hops))

    return


class Batcher:
    def __init__(self,
                 batch_size: int,
                 nb_examples: int,
                 nb_epochs: int,
                 random_state: Optional[np.random.RandomState]):
        self.batch_size = batch_size
        self.nb_examples = nb_examples
        self.nb_epochs = nb_epochs
        self.random_state = random_state

        size = self.nb_epochs * self.nb_examples
        self.curriculum = np.zeros(size, dtype=np.int32)

        for epoch_no in range(nb_epochs):
            start, end = epoch_no * nb_examples, (epoch_no + 1) * nb_examples
            if self.random_state is not None:
                self.curriculum[start: end] = self.random_state.permutation(nb_examples)
            else:
                self.curriculum[start: end] = np.arange(nb_examples)

        self.batches = make_batches(self.curriculum.shape[0], self.batch_size)
        self.nb_batches = len(self.batches)

    def get_batch(self,
                  batch_start: int,
                  batch_end: int) -> np.ndarray:
        return self.curriculum[batch_start:batch_end]

# takes in the list of facts [F,3]
# gives back the list of the embeddings corresponding to the relations of each fact [F,E]
def encode_relation(facts: List[Fact], # [F,3]: [(s,r,o), (s,r,o), ..., (s,r,o)]
                    relation_embeddings: Tensor,
                    relation_to_idx: Dict[str, int],
                    device: Optional[torch.device] = None) -> Tensor:
    indices_np = np.array([relation_to_idx[r] for _, r, _ in facts], dtype=np.int64)
    indices = torch.tensor(indices_np, dtype=torch.long, device=device)
    res = F.embedding(indices, relation_embeddings)
    return res

# takes in the list of facts [F,3]
# gives back the list of the embeddings corresponding to the subject and object of each fact [F,E],[F,E]
def encode_arguments(facts: List[Fact],
                     entity_embeddings: Tensor,
                     entity_to_idx: Dict[str, int],
                     device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    indices_np = np.array([[entity_to_idx[s], entity_to_idx[o]] for s, _, o in facts], dtype=np.int64)
    indices = torch.tensor(indices_np, dtype=torch.long, device=device)
    emb = F.embedding(indices, entity_embeddings)
    return emb[:, 0, :], emb[:, 1, :]

# takes in the list of facts [F,3]
# gives back the list of the embeddings corresponding to all entities that are in the facts as either
# subject or object
# [N,E], where N is the number of different entities in the facts
def encode_entities(facts: List[Fact],
                    entity_embeddings: Tensor,
                    entity_to_idx: Dict[str, int],
                    device: Optional[torch.device]) -> Tensor:
    indices_lst = sorted({entity_to_idx[e] for s, r, o in facts for e in {s, o}})
    indices = torch.tensor(indices_lst, dtype=torch.long, device=device)
    emb = F.embedding(indices, entity_embeddings)
    return emb


def main():

    debug = False

    train_path = test_path = join(dirname(dirname(abspath(__file__))),'data', 'clutrr-emnlp', 'data_db9b8f04', '1.2,1.3,1.4_train.csv') # "data/clutrr-emnlp/data_test/64.csv"
    test_path1 = join(dirname(dirname(abspath(__file__))),'data', 'clutrr-emnlp', 'data_db9b8f04', '1.10_test.csv')
    # test_path2 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.2_test.csv')
    test_path3 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.3_test.csv')
    test_path4 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.4_test.csv')
    test_path5 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.5_test.csv')
    test_path6 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.6_test.csv')
    test_path7 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.7_test.csv')
    test_path8 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.8_test.csv')
    test_path9 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.9_test.csv')
    test_paths = [test_path1, test_path3, test_path4, test_path5, test_path6, test_path7, test_path8, test_path9]
    # test_path2,
    # model params

    # the size of the embedding of each atom and relationship
    embedding_size = 20
    # when proving the body of a rule, we consider the k best substitutions for each variable
### OTHER
    k_max = 10 #10, 5 is suggested
    # how many times to reformulate the goal(s) --> bigger for bigger graph: this is for training
    max_depth = 2
    # how many times to reformulate the goal(s): this is for testing --> this depth can be bigger than for training
    test_max_depth = 2

    # the shape of the reformulation:
    # 2: goal(X,Z) -> p(X,Y), q(Y,Z) (2 elements in the body)
    # 1: goal(X,Z) -> r(X,Z) (1 element in the body)
    # 1R: goal(X,Z) -> s(Z,X) (variables in reversed order)
    # if we have multiple in the array, that means at each reformulation step we actually reformulate the same goal
    # multiple times according to the elements in the array (so we have more choice to get to a good proof)
    hops_str = ['2', '2', '2'] # ['2', '2', '1R']


    # training params

### OTHER
    nb_epochs = 20 # 100, 5 is suggested
### OTHER
    learning_rate = 0.01 #0.1 is suggested
    # training batch size
    batch_size = 32
    # testing batch size --> this can be smaller than for training
    test_batch_size = batch_size # could be other as well

    optimizer_name = 'adagrad' # choices = ['adagrad', 'adam', 'sgd']

    seed = 1 # int

    # how often you want to evaluate
    evaluate_every = 128 # int
    evaluate_every_batches = None # int

    # whether you want to regularize
    #argparser.add_argument('--N2', action='store', type=float, default=None)
    #argparser.add_argument('--N3', action='store', type=float, default=None)
    #argparser.add_argument('--entropy', '-E', action='store', type=float, default=None)

    scoring_type = 'concat' # choices = ['concat', 'min']

    # how to get the score combined from the conjunction of the parts of the body in a rule
    # e.g. goal(X,Y) :- p(X,Z), q(Z,Y) --> how do we combine the scores of p and q
    # I can just keep min, it works well, and later add the others potentially
    tnorm_name = 'min' # choices = ['min', 'prod', 'mean']
    # which function to use for reformulating a goal to subgoals
    # I can just keep linear for the initial version, and if it works, I can add more
    reformulator_name = 'linear' # choices = ['static', 'linear', 'attentive', 'memory', 'ntp'] --> deleted code part

    nb_gradient_accumulation_steps = 1 # int

#### IGNORE --> code as well
    gntp_R = None # int

    slope = 1.0 # float
    init_size = 1.0 # float

    init_type = 'random'  # 'uniform'
    ref_init_type = 'random'

#### IGNORE
    #argparser.add_argument('--fix-relations', '--FR', action='store_true', default=False)
    is_fixed_relations = False
    # whether you want to train on the smallest graph first (it's easier to train on them)
    start_simple = None # int

    is_debug = False

    load_path = None # str
    save_path = None # str

    # took out predicates from the code, no need for them (they take only 1 argument, but an edge in PyG always takes 2)
    #is_predicate = False

# IGNORED variables:
    #N2_weight = args.N2
    #N3_weight = args.N3
    #entropy_weight = args.entropy

    #is_fixed_relations = args.fix_relations


    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Initializing data and embeddings

    data = DataParser(train_path=train_path, test_paths=test_paths)
    entity_lst, relation_lst = data.entity_lst, data.relation_lst
    # relation_lst = ['father', 'son', 'wife', 'husband', 'uncle', 'grandfather', 'grandmother', 'daughter']

    test_relation_lst = ["aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law", "granddaughter",
                         "grandfather", "grandmother", "grandson", "mother", "mother-in-law", "nephew", "niece",
                         "sister", "son", "son-in-law", "uncle"]

    nb_entities = len(entity_lst)
    nb_relations = len(relation_lst)

    # entity_to_idx = {e: i for i, e in enumerate(entity_lst)}
    relation_to_idx = data.edge_types_to_class  # {r: i for i, r in enumerate(relation_lst)}

    kernel = GaussianKernel(slope=slope)

    # [N,E], where N is the number of different entities (nb_entities)
    entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True).to(device)
    nn.init.uniform_(entity_embeddings.weight, -1.0, 1.0)
    entity_embeddings.requires_grad = False

    relation_embeddings = nn.Embedding(nb_relations, embedding_size, sparse=True).to(device)
    #if is_fixed_relations is True:
    relation_embeddings.requires_grad = False

    if init_type in {'uniform'}:
        nn.init.uniform_(relation_embeddings.weight, -1.0, 1.0)

    relation_embeddings.weight.data *= init_size

    # the model that lets you look up in the KB
    model = BatchNeuralKB(kernel=kernel, scoring_type=scoring_type).to(device)
    memory: Dict[int, MemoryReformulator.Memory] = {}

    # generates a reformulator of type given as an argument
    def make_hop(s: str) -> Tuple[BaseReformulator, bool]:
        nonlocal memory
        if s.isdigit():
            nb_hops, is_reversed = int(s), False
        else:
            nb_hops, is_reversed = int(s[:-1]), True
        res = None

        if reformulator_name in {'linear'}:
            res = LinearReformulator(nb_hops, embedding_size, init_name=ref_init_type)

        assert res is not None
        return res.to(device), is_reversed

    hops_lst = [make_hop(s) for s in hops_str]  # hops_str = [2,2,1R]

    # "model" is a neural KB for checking whether the facts are true or not
    # hoppy is the model that does the reasoning, using the neural KB
    hoppy = BatchHoppy(model=model, k=k_max, depth=max_depth, tnorm_name=tnorm_name,
                       hops_lst=hops_lst).to(device)
    """
    def scoring_function(instances_batch: List[Instance],
                         relation_lst: List[str],
                         is_train: bool = False,
                         _depth: Optional[int] = None) -> Tuple[Tensor, List[Tensor]]:

        rel_emb_lst: List[Tensor] = []
        arg1_emb_lst: List[Tensor] = []
        arg2_emb_lst: List[Tensor] = []

        story_rel_lst: List[Tensor] = []
        story_arg1_lst: List[Tensor] = []
        story_arg2_lst: List[Tensor] = []

        embeddings_lst: List[Tensor] = []

        label_lst: List[int] = []

        for i, instance in enumerate(instances_batch):

            story, target = instance.story, instance.target
            s, r, o = target  # 1 target relation

            # the relation embeddings from all facts (==story)
            # [F,E], where F is the number of facts
            story_rel = encode_relation(story, relation_embeddings.weight, relation_to_idx, device)
            # the subject,object embeddings from all facts (==story)
            # [F,E],[F,E]
            story_arg1, story_arg2 = encode_arguments(story, entity_embeddings.weight, entity_to_idx, device)

            # the subject,object embeddings from all facts (==story) as a list of all different entity embeddings
            # [N,E], where N is the number of different entities in the facts
            embeddings = encode_entities(story, entity_embeddings.weight, entity_to_idx, device)

            # target subject (s), object(o) paired with all possible relations from the relation_lst (x)
            # [R,3] where R is the number of relations
            target_lst: List[Tuple[str, str, str]] = [(s, x, o) for x in relation_lst]

            assert len(target_lst) == len(test_relation_lst)

            # true_predicate = rel_to_predicate[r]
            # label_lst += [int(true_predicate == rel_to_predicate[r]) for r in relation_lst]

            # for each instance in the batch we are iterating through: [0,0,...,0,1,0,0,...,0] list
            # where 1 is at the relation that is the target relation
            label_lst += [int(tr == r) for tr in relation_lst]

            # relation embeddings of the target subject,object paired with all possible relations - R relations
            # [R,E]
            rel_emb = encode_relation(target_lst, relation_embeddings.weight, relation_to_idx, device)
            # target subject,object embeddings
            # [R,E],[R,E] (actually all will be the same as the subject, object doesn't change just the relations)
            arg1_emb, arg2_emb = encode_arguments(target_lst, entity_embeddings.weight, entity_to_idx, device)

            batch_size = rel_emb.shape[0]  # R - the number of relations --> B from now on
            fact_size = story_rel.shape[0]  # F - the number of facts in "story"
            entity_size = embeddings.shape[0]  # N - the number of different entities in the facts (=="story")

            # [B, E]
            # at each instance in the batch we add [B,E] to it (B==R)
            # --> rel_emb_lst will become [batch,B,E], where batch is number of instances in batch
            rel_emb_lst += [rel_emb]
            arg1_emb_lst += [arg1_emb]
            arg2_emb_lst += [arg2_emb]

            # [B, F, E]
            # repeat the same facts for each rel subst in instance of current batch
            # (1 batch will be all possible relation substitutions for the target relation)
            # --> story_rel_lst will become [batch,B,F,E], where batch is number of instances in batch
            story_rel_lst += [story_rel.view(1, fact_size, -1).repeat(batch_size, 1, 1)]
            story_arg1_lst += [story_arg1.view(1, fact_size, -1).repeat(batch_size, 1, 1)]
            story_arg2_lst += [story_arg2.view(1, fact_size, -1).repeat(batch_size, 1, 1)]

            # [B, N, E]
            # repeat the same entity embeddings for each instance in batch (== for each relation)
            # (1 batch will be all possible relation substitutions for the target relation)
            # --> embeddings_lst will become [batch,B,N,E], where batch is number of instances in batch
            embeddings_lst += [embeddings.view(1, entity_size, -1).repeat(batch_size, 1, 1)]

        def cat_pad(t_lst: List[Tensor]) -> Tuple[Tensor, Tensor]:
            # t: [B,F,E] --> t.shape[1] == F --> F possibly different each time
            # OR
            # t: [B,N,E] --> t.shape[1] == N --> N possibly different each time
            lengths: List[int] = [t.shape[1] for t in t_lst]
            max_len: int = max(lengths)

            # pad the _t tensor: [B,F,E] (which is all t in t_lst) with embeddings of 0 where F is smaller than max_len
            # OR
            # pad the _t tensor: [B,N,E] (which is all t in t_lst) with embeddings of 0 where N is smaller than max_len
            def my_pad(_t: Tensor, pad: List[int]) -> Tensor:
                return torch.transpose(F.pad(torch.transpose(_t, 1, 2), pad=pad), 1, 2)

            # [batch*B,F,E], where each F is the same (sometimes padded with extra 0 embeddings to reach that)
            # OR
            # [batch*B,N,E], where each N is the same (sometimes padded with extra 0 embeddings to reach that)
            res_t: Tensor = torch.cat([my_pad(t, pad=[0, max_len - lengths[idx]]) for idx, t in enumerate(t_lst)],
                                      dim=0)
            # [batch*B], the number of facts (before padding) in an instance
            # OR
            # [batch*B], the number of entities (before padding) in an instance
            res_l: Tensor = torch.tensor([t.shape[1] for t in t_lst for _ in range(t.shape[0])],
                                         dtype=torch.long, device=device)
            return res_t, res_l  # [batch*B,F,E],[batch*B] OR [batch*B,N,E],[batch*B]

        # [batch*B,E], where batch is number of batches & B is batch_size that is number of relations
        rel_emb = torch.cat(rel_emb_lst, dim=0)
        arg1_emb = torch.cat(arg1_emb_lst, dim=0)
        arg2_emb = torch.cat(arg2_emb_lst, dim=0)

        # story_rel,story_arg1,story_arg2: [batch*B,F,E], where each F is the same (sometimes padded with extra 0 embeddings to reach that)
        # nb_facts: [batch*B], the number of facts (before padding) in each instance
        story_rel, nb_facts = cat_pad(story_rel_lst)  # story_rel_lst: [batch,B,F,E]
        story_arg1, _ = cat_pad(story_arg1_lst)
        story_arg2, _ = cat_pad(story_arg2_lst)
        facts = [story_rel, story_arg1, story_arg2]  # [3,batch*B,F,E]

        # _embeddings: [batch*B,N,E], where each N is the same (sometimes padded with extra 0 embeddings to reach that)
        # nb_embeddings: [batch*B], the number of entities (before padding) in each instance
        _embeddings, nb_embeddings = cat_pad(embeddings_lst)

        max_depth_ = hoppy.depth
        if not is_train and test_max_depth is not None:
            hoppy.depth = test_max_depth

        if _depth is not None:
            hoppy.depth = _depth

        # TODO: write this with HeteroData
        # trying hoppy.prove instead
        # scores = hoppy.score(rel_emb, arg1_emb, arg2_emb, facts, nb_facts, _embeddings, nb_embeddings)
        scores = hoppy.prove(rel_emb, arg1_emb, arg2_emb, facts, nb_facts, _embeddings, nb_embeddings, hoppy.depth)

        if not is_train and test_max_depth is not None:
            hoppy.depth = max_depth_

        if _depth is not None:
            hoppy.depth = max_depth_  # ??

        return scores, [rel_emb, arg1_emb, arg2_emb]
    """

    def scoring_function_CTP(graph_data: HeteroData,
                             relation_to_class: Dict[str, int],
                             is_train: bool = False,
                             _depth: Optional[int] = None) -> Tuple[Tensor, List[Tensor]]:

        # instances_batch: List[Instance],
        #             relation_lst: List[str],
        #             is_train: bool = False,
        #             _depth: Optional[int] = None) -> Tuple[Tensor, List[Tensor]]:

        # TODO: make batching, because "facts_rel = facts_rel.view(1, fact_size, -1).repeat(batch_size, 1, 1)"
        # TODO: runs out of memory
        # [N,E], where N is the number of different entities
        # entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True).to(device)
        # nn.init.uniform_(entity_embeddings.weight, -1.0, 1.0)
        # entity_embeddings.requires_grad = False

        # rel_embeddings = nn.Embedding(nb_relations, embedding_size, sparse=True).to(device)
        # nn.init.uniform_(rel_embeddings.weight, -1.0, 1.0)
        # rel_embeddings.requires_grad = False

        # example input:
        # edge_index_dict = {
        #    'parent': torch.LongTensor([[0, 1], [1, 0]]),
        #    'dog': torch.LongTensor([[2, 3, 0], [1, 0, 1]])
        # }
        # x_dict = {'entity': torch.rand(5, 32)}
        # rel_embeddings = torch.rand(2, 16)
        # relation_to_class = {'parent': 0, 'dog': 1}

        #relation_to_class = graph_data.edge_types_to_class
        #train_graph = graph_data.train_graph

        edge_index_dict = graph_data.edge_index_dict
        x_dict = graph_data.x_dict
        edge_label_index = graph_data['entity', 'target', 'entity'].edge_index
        # torch.LongTensor([[1, 2, 3], [0, 2, 1]])

        ## encoding targets
        # rel = torch.tile(relation_embeddings, (edge_label_index.shape[1], 1))
        rel_idx = torch.tile(torch.arange(nb_relations), (edge_label_index.shape[1], 1))
        rel = relation_embeddings(rel_idx) # [nb_targets*R,E]
        arg1_idx = torch.tile(edge_label_index[0, :], (nb_relations, 1)).T.flatten()  # relation_embeddings.shape[0]
        # [nb_targets*R,E]
        arg1 = entity_embeddings(arg1_idx) # x_dict['entity'][arg1_idx]
        arg2_idx = torch.tile(edge_label_index[1, :], (nb_relations, 1)).T.flatten()
        # [nb_targets*R,E]
        arg2 = entity_embeddings(arg2_idx) # x_dict['entity'][arg2_idx]

        ## Encoding facts
        # F: total number of facts
        facts_arg1 = torch.cat([entity_embeddings(edge_index[0, :]) for edge_index in edge_index_dict.values()]) # [F,E]
        # torch.cat([x_dict['entity'][edge_index[0, :], :] for edge_index in edge_index_dict.values()])
        facts_arg2 = torch.cat([entity_embeddings(edge_index[1, :]) for edge_index in edge_index_dict.values()]) # [F,E]
        # torch.cat([x_dict['entity'][edge_index[1, :], :] for edge_index in edge_index_dict.values()])

        #for key, edge_index in edge_index_dict.items():
        #    for edge_type in [key] * edge_index.shape[1]:
        #        # print(edge_type)
        #        print(key[1])

        rel_index = []
        for key, edge_index in edge_index_dict.items():
            if key[1] != 'target':
                for edge_type in [key[1]] * edge_index.shape[1]:
                    rel_index.append(relation_to_class[edge_type])
        rel_index = torch.tensor(rel_index, device=device)

        #rel_index = [relation_to_class[edge_type] for key, edge_index in edge_index_dict.items() for edge_type in
        #             [key[1]] * edge_index.shape[1]]
        facts_rel = relation_embeddings(rel_index)  # [F,E]
        print(f"facts_rel shape should be [F,E], and it is: {facts_rel.shape}")

        batch_size = rel.shape[0]  # nb_targets*R
        fact_size = facts_rel.shape[0]  # F - the number of facts in "story"
        entity_size = nb_entities  # N - the number of different entities in the facts (=="story")

        # [B,F,E]
        facts_rel = facts_rel.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        facts_arg1 = facts_arg1.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        facts_arg2 = facts_arg2.view(1, fact_size, -1).repeat(batch_size, 1, 1)

        # [3,B,F,E]
        facts = [facts_rel, facts_arg1, facts_arg2]  # pass to prove function

        # entity_emb = x_dict['entity']

        # rel_emb_lst: List[Tensor] = []
        # arg1_emb_lst: List[Tensor] = []
        # arg2_emb_lst: List[Tensor] = []

        # story_rel_lst: List[Tensor] = []
        # story_arg1_lst: List[Tensor] = []
        # story_arg2_lst: List[Tensor] = []

        # embeddings_lst: List[Tensor] = []

        # label_lst: List[int] = []

        #pred = model(train_data.x_dict, train_data.edge_index_dict, train_data['entity', 'target', 'entity'].edge_index)
        #pred = pred.clamp(min=0, max=1)
        label_lst = torch.zeros(graph_data['entity', 'target', 'entity'].edge_index.shape[1]*nb_relations, device=device)
        for i in range(len(graph_data['entity', 'target', 'entity'].edge_label)):
            zero_idx = i * nb_relations
            class_num = graph_data['entity', 'target', 'entity'].edge_label[i]
            label_lst[zero_idx + class_num] = 1
        # TODO: one_hot-tal megcsinalni a targetet

        embeddings_lst = [entity_embeddings]

        """
        def cat_pad(t_lst: List[Tensor]) -> Tuple[Tensor, Tensor]:
            # t: [B,F,E] --> t.shape[1] == F --> F possibly different each time
            # OR
            # t: [B,N,E] --> t.shape[1] == N --> N possibly different each time
            lengths: List[int] = [t.shape[1] for t in t_lst]
            max_len: int = max(lengths)

            # pad the _t tensor: [B,F,E] (which is all t in t_lst) with embeddings of 0 where F is smaller than max_len
            # OR
            # pad the _t tensor: [B,N,E] (which is all t in t_lst) with embeddings of 0 where N is smaller than max_len
            def my_pad(_t: Tensor, pad: List[int]) -> Tensor:
                return torch.transpose(F.pad(torch.transpose(_t, 1, 2), pad=pad), 1, 2)

            # [batch*B,F,E], where each F is the same (sometimes padded with extra 0 embeddings to reach that)
            # OR
            # [batch*B,N,E], where each N is the same (sometimes padded with extra 0 embeddings to reach that)
            res_t: Tensor = torch.cat([my_pad(t, pad=[0, max_len - lengths[idx]]) for idx, t in enumerate(t_lst)],
                                      dim=0)
            # [batch*B], the number of facts (before padding) in an instance
            # OR
            # [batch*B], the number of entities (before padding) in an instance
            res_l: Tensor = torch.tensor([t.shape[1] for t in t_lst for _ in range(t.shape[0])],
                                         dtype=torch.long, device=device)
            return res_t, res_l  # [batch*B,F,E],[batch*B] OR [batch*B,N,E],[batch*B]
        """

        # [batch*B,E], where batch is number of batches & B is batch_size that is number of relations
        # rel_emb = torch.cat(rel_emb_lst, dim=0)
        # arg1_emb = torch.cat(arg1_emb_lst, dim=0)
        # arg2_emb = torch.cat(arg2_emb_lst, dim=0)

        # story_rel,story_arg1,story_arg2: [batch*B,F,E], where each F is the same (sometimes padded with extra 0 embeddings to reach that)
        # nb_facts: [batch*B], the number of facts (before padding) in each instance
        # story_rel, nb_facts = cat_pad(story_rel_lst)  # story_rel_lst: [batch,B,F,E]
        # story_arg1, _ = cat_pad(story_arg1_lst)
        # story_arg2, _ = cat_pad(story_arg2_lst)
        # facts = [story_rel, story_arg1, story_arg2]  # [3,batch*B,F,E]

        # _embeddings: [batch*B,N,E], where each N is the same (sometimes padded with extra 0 embeddings to reach that)
        # nb_embeddings: [batch*B], the number of entities (before padding) in each instance
        # _embeddings, nb_embeddings = cat_pad(embeddings_lst)

        nb_facts = facts[0].shape[0]

        max_depth_ = hoppy.depth
        if not is_train and test_max_depth is not None:
            hoppy.depth = test_max_depth

        if _depth is not None:
            hoppy.depth = _depth

        # TODO: write this with HeteroData
        # trying hoppy.prove instead
        # scores = hoppy.score(rel_emb, arg1_emb, arg2_emb, facts, nb_facts, _embeddings, nb_entities)
        scores = hoppy.prove(rel, arg1, arg2, facts, nb_facts, embeddings_lst, nb_entities, hoppy.depth)

        if not is_train and test_max_depth is not None:
            hoppy.depth = max_depth_

        if _depth is not None:
            hoppy.depth = max_depth_  # ??

        return scores, [rel, arg1, arg2]  # [rel_emb, arg1_emb, arg2_emb]

    """
    def evaluate(instances: List[Instance],
                 path: str,
                 sample_size: Optional[int] = None) -> float:
        res = 0.0
        if len(instances) > 0:
            res = accuracy(scoring_function=scoring_function,
                           instances=instances,
                           sample_size=sample_size,
                           relation_lst=test_relation_lst,
                           batch_size=test_batch_size,
                           relation_to_predicate=None,
                           is_debug=is_debug)
            logger.info(f'Test Accuracy on {path}: {res:.6f}')
        return res
    """

    def evaluate_CTP(graph_data: HeteroData,
                     relation_to_class: Dict[str, int],
                     path: str) -> float:
        res = accuracy(scoring_function=scoring_function_CTP,
                       graph_data=graph_data,
                       relation_to_class=relation_to_class,
                       relation_lst=relation_lst,
                       batch_size=1,
                       is_debug=is_debug)
        logger.info(f'Test Accuracy on {path}: {res:.6f}')
        return res

    loss_function = nn.BCELoss()

    params_lst = [p for p in hoppy.parameters() if not torch.equal(p, entity_embeddings.weight)]

    if is_fixed_relations is False:
        params_lst += relation_embeddings.parameters()

    params = nn.ParameterList(params_lst).to(device)

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    for tensor in params_lst:
        logger.info(f'\t{tensor.size()}\t{tensor.device}')

    # TODO: set just adam for example, no need for all these choices
    optimizer_factory = {
        'adagrad': lambda arg: optim.Adagrad(arg, lr=learning_rate),
        'adam': lambda arg: optim.Adam(arg, lr=learning_rate),
        'sgd': lambda arg: optim.SGD(arg, lr=learning_rate)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name](params)

    global_step = 0
    relation_to_class = data.edge_types_to_class

    for epoch_no in range(1, nb_epochs + 1):

        train_data, is_simple = data.train_graph, False
        # if start_simple is not None and epoch_no <= start_simple:
        #    training_set = [ins for ins in training_set if len(ins.story) == 2]
        #    is_simple = True
        #    logger.info(f'{len(data.train)} → {len(training_set)}')

        # batcher = Batcher(batch_size=batch_size, nb_examples=len(training_set), nb_epochs=1, random_state=random_state)

        nb_batches = 1  # len(batcher.batches)
        epoch_loss_values = []

        """
        counter = 0
        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, start=1):
            counter += 1
            if debug:
                if counter > 1:
                    break
            global_step += 1
        """

        # getting current batch from the training set
        # indices_batch = batcher.get_batch(batch_start, batch_end)
        # instances_batch = [training_set[i] for i in indices_batch]

        # label_lst: list of 1s and 0s indicating which query (==target) relation/predicate is where in the test_predicate_lst
        # DONE: take out predicates, no need for them (they take only 1 argument, but an edge in PyG always takes 2)

        # label_lst: List[int] = [int(ins.target[1] == tr) for ins in instances_batch for tr in test_relation_lst]

        label_lst = torch.zeros(train_data['entity', 'target', 'entity'].edge_index.shape[1]*nb_relations, device=device)
        for i in range(len(train_data['entity', 'target', 'entity'].edge_label)):
            zero_idx = i * nb_relations
            class_num = train_data['entity', 'target', 'entity'].edge_label[i]
            label_lst[zero_idx + class_num] = 1
        # TODO: one_hot-tal megcsinalni a targetet

        labels = torch.tensor(label_lst, dtype=torch.float32, device=device)

        # returns scores of what??
        scores, query_emb_lst = scoring_function_CTP(train_data,
                                                     relation_to_class,
                                                     is_train=True,
                                                     _depth=1 if is_simple else None)

        loss = loss_function(scores, labels)

        factors = [hoppy.factor(e) for e in query_emb_lst]

        loss_value = loss.item()
        epoch_loss_values += [loss_value]

        if nb_gradient_accumulation_steps > 1:
            loss = loss / nb_gradient_accumulation_steps

        loss.backward()

        if nb_gradient_accumulation_steps == 1 or global_step % nb_gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch 1/{nb_batches}\tLoss {loss_value:.4f}')
        # logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.4f}')

        #if evaluate_every_batches is not None:
        #    if global_step % evaluate_every_batches == 0:
        #        for test_path in test_paths:
        #            evaluate_CTP(instances=data.test[test_path], path=test_path)

        if epoch_no % evaluate_every == 0:
            for test_path in test_paths:
                evaluate_CTP(graph_data=data.test_graphs[test_path],
                             relation_to_class=relation_to_class,
                             path=test_path)
                # evaluate_CTP(instances=data.test[test_path], path=test_path)

            # TODO: don't need for now, but we may look at it later
            if is_debug is True:
                with torch.no_grad():
                    show_rules(model=hoppy, kernel=kernel, relation_embeddings=relation_embeddings,
                               relation_to_idx=relation_to_idx, device=device)

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)

        slope = kernel.slope.item() if isinstance(kernel.slope, Tensor) else kernel.slope
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}\tSlope {slope:.4f}')

    import time
    start = time.time()

    for test_path in test_paths:
        evaluate_CTP(graph_data=data.test_graphs[test_path],
                     relation_to_class=relation_to_class,
                     path=test_path)
        # evaluate(instances=data.test[test_path], path=test_path)

    end = time.time()
    logger.info(f'Evaluation took {end - start} seconds.')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    print(' '.join(sys.argv))
    main()
