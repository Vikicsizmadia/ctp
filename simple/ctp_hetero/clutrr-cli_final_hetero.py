#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join, dirname, abspath
import sys

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor

from ctp.util import make_batches

from make_heterodata_final_CTP import DataParserCTP
from evaluation_heterodata import accuracy
from util_hetero_final import get_neighbours
from kb_final_hetero import BatchNeuralKB
from model_hetero_final import BatchHoppy
from gaussian_final import GaussianKernel

from ctp.reformulators import BaseReformulator
from ctp.reformulators import LinearReformulator
from ctp.reformulators import MemoryReformulator

from typing import List, Tuple, Dict, Optional

from torch_geometric.data import HeteroData

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=256, precision=4, suppress=True, threshold=sys.maxsize)

torch.set_num_threads(multiprocessing.cpu_count())

class BatcherHetero:
    def __init__(self,
                 batch_size: int,
                 nb_examples: int, # number of targets
                 random_state: Optional[np.random.RandomState]):
        self.batch_size = batch_size
        self.nb_examples = nb_examples
        self.random_state = random_state

        self.curriculum = np.zeros(self.nb_examples, dtype=np.int32)

        if self.random_state is not None:
            self.curriculum = self.random_state.permutation(nb_examples)
        else:
            self.curriculum = np.arange(nb_examples)

        self.batches = make_batches(self.curriculum.shape[0], self.batch_size)
        self.nb_batches = len(self.batches)

    def get_batch(self,
                  batch_start: int,
                  batch_end: int) -> np.ndarray:
        return self.curriculum[batch_start:batch_end]


def main():

    train_path = join(dirname(dirname(dirname(abspath(__file__)))),'data', 'clutrr-emnlp', 'data_test', '64.csv')
    test_path1 = join(dirname(dirname(dirname(abspath(__file__)))),'data', 'clutrr-emnlp', 'data_db9b8f04', '1.10_test.csv')
    test_path2 = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.2_test.csv')
    test_path3 = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.3_test.csv')
    test_path4 = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.4_test.csv')
    test_path5 = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.5_test.csv')
    test_path6 = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.6_test.csv')
    test_path7 = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.7_test.csv')
    test_path8 = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.8_test.csv')
    test_path9 = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.9_test.csv')
    test_paths = [test_path1, test_path2, test_path3, test_path4, test_path5, test_path6, test_path7, test_path8, test_path9]

    # model params

    # the size of the embedding of each atom and relationship
    embedding_size = 20
    # when proving the body of a rule, we consider the k best substitutions for each variable
### OTHER
    k_max = 5  # 10, 5 is suggested
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
    hops_str = ['2', '2', '2']  # ['2', '2', '1R']


    # training params

### OTHER0
    nb_epochs = 50  # 100, 5 is suggested
### OTHER
    learning_rate = 0.1  # 0.1 is suggested
    # training batch size
    batch_size = 4  # 32
    # testing batch size --> this can be smaller than for training
    test_batch_size = 2  # could be other as well

    optimizer_name = 'adagrad'  # choices = ['adagrad', 'adam', 'sgd']

    seed = 1  # int

    # how often you want to evaluate
    evaluate_every = None  # int 128
    evaluate_every_batches = None  # None, int 10

    # whether you want to regularize
    #argparser.add_argument('--N2', action='store', type=float, default=None)
    #argparser.add_argument('--N3', action='store', type=float, default=None)
    #argparser.add_argument('--entropy', '-E', action='store', type=float, default=None)

    scoring_type = 'concat'  # choices = ['concat', 'min']

    # how to get the score combined from the conjunction of the parts of the body in a rule
    # e.g. goal(X,Y) :- p(X,Z), q(Z,Y) --> how do we combine the scores of p and q
    # I can just keep min, it works well, and later add the others potentially
    tnorm_name = 'min'  # choices = ['min', 'prod', 'mean']
    # which function to use for reformulating a goal to subgoals
    # I can just keep linear for the initial version, and if it works, I can add more
    reformulator_name = 'linear'  # choices = ['static', 'linear', 'attentive', 'memory', 'ntp'] --> deleted code part

    nb_gradient_accumulation_steps = 1 # int

    slope = 1.0 # float
    init_size = 1.0 # float

    init_type = 'random'  # 'uniform'
    ref_init_type = 'random'

#### IGNORE
    # argparser.add_argument('--fix-relations', '--FR', action='store_true', default=False)
    is_fixed_relations = False
    # whether you want to train on the smallest graph first (it's easier to train on them)
    start_simple = None  # int

    is_debug = False

    load_path = None  # str
    save_path = None  # str

# IGNORED variables:
    #N2_weight = args.N2
    #N3_weight = args.N3
    #entropy_weight = args.entropy

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Initializing data and embeddings

    data = DataParserCTP(train_path=train_path, test_paths=test_paths)
    entity_lst, relation_lst = data.entity_lst, data.relation_lst
    # relation_lst = ['father', 'son', 'wife', 'husband', 'uncle', 'grandfather', 'grandmother', 'daughter']

    test_relation_lst = ["aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law", "granddaughter",
                         "grandfather", "grandmother", "grandson", "mother", "mother-in-law", "nephew", "niece",
                         "sister", "son", "son-in-law", "uncle"]

    nb_entities = len(entity_lst)
    nb_relations = len(relation_lst)

    # entity_to_idx = {e: i for i, e in enumerate(entity_lst)}
    # relation_to_idx = data.edge_types_to_class  # {r: i for i, r in enumerate(relation_lst)}

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

    # "model" is a neural KB for getting similarity scores for queries to the facts
    # hoppy is the model that does the reasoning, using the neural KB
    hoppy = BatchHoppy(model=model, k=k_max, depth=max_depth, tnorm_name=tnorm_name, hops_lst=hops_lst).to(device)

    def scoring_function_CTP(graph_data: HeteroData,
                             relation_to_class: Dict[str, int],
                             entity_lst: List[int],
                             is_train: bool = False,
                             _depth: Optional[int] = None) -> Tuple[Tensor, List[Tensor]]:

        edge_index_dict = graph_data.edge_index_dict
        edge_label_index = graph_data['entity', 'target', 'entity'].edge_index

        # Encoding targets

        # [nb_targets*R,E]
        rel_idx = torch.tile(torch.arange(nb_relations), (edge_label_index.shape[1], 1)).flatten()
        rel = relation_embeddings(rel_idx)
        arg1_idx = torch.tile(edge_label_index[0, :], (nb_relations, 1)).T.flatten()
        arg1 = entity_embeddings(arg1_idx)
        arg2_idx = torch.tile(edge_label_index[1, :], (nb_relations, 1)).T.flatten()
        arg2 = entity_embeddings(arg2_idx)

        # Encoding facts

        # F: total number of facts
        # necessary that the target edge type is the last edge type
        # [F,E]
        facts_arg1 = torch.cat([entity_embeddings(edge_index[0, :]) for edge_index in list(edge_index_dict.values())[:-1]])
        facts_arg2 = torch.cat([entity_embeddings(edge_index[1, :]) for edge_index in list(edge_index_dict.values())[:-1]])

        rel_index = []
        for key, edge_index in edge_index_dict.items():
            if key[1] != 'target':  # we don't want to include the target edges in the facts
                rel_index.extend([relation_to_class[key[1]]] * edge_index.shape[1])
        rel_index = torch.tensor(rel_index, dtype=torch.long, device=device)

        facts_rel = relation_embeddings(rel_index)  # [F,E]

        batch_size = rel.shape[0]  # nb_targets*R = B

        entity_size = len(entity_lst)  # N - the number of different entities in the facts
        entity_tensor = torch.tensor(entity_lst, dtype=torch.long, device=device)
        embeddings = entity_embeddings(entity_tensor)

        # [B, N, E]
        # repeat the same entity embeddings for each instance in batch (== for each relation)
        # (1 batch will be all possible relation substitutions for the target relation)
        # --> embeddings_lst will become [batch_size,N,E], where batch is number of instances in batch
        embeddings = embeddings.view(1, entity_size, -1).repeat(batch_size, 1, 1)

        # [3,F,E]
        facts = [facts_rel, facts_arg1, facts_arg2]  # pass to prove function

        max_depth_ = hoppy.depth
        if not is_train and test_max_depth is not None:
            hoppy.depth = test_max_depth

        if _depth is not None:
            hoppy.depth = _depth

        # [nb_targets*R] = [B]
        scores = hoppy.prove(rel, arg1, arg2, facts, embeddings, hoppy.depth)

        if not is_train and test_max_depth is not None:
            hoppy.depth = max_depth_

        if _depth is not None:
            hoppy.depth = max_depth_

        return scores, [rel, arg1, arg2]  # [rel_emb, arg1_emb, arg2_emb]

    def evaluate_CTP(graph_data: HeteroData,
                     relation_to_class: Dict[str, int],
                     path: str) -> float:
        res = accuracy(scoring_function=scoring_function_CTP,
                       graph_data=graph_data,
                       relation_to_class=relation_to_class,
                       relation_lst=relation_lst,
                       batch_size=test_batch_size)
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

        targets = train_data['entity', 'target', 'entity'].edge_index
        target_labels = train_data['entity', 'target', 'entity'].edge_label

        batcher = BatcherHetero(batch_size=batch_size, nb_examples=targets.shape[1], random_state=random_state)

        nb_batches = len(batcher.batches)
        epoch_loss_values = []

        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, start=1):
            global_step += 1

            # TODO: put all this into the Batcher and get current_data from that
            # getting current batch from the training set
            indices_batch = batcher.get_batch(batch_start, batch_end)
            node_ids = torch.cat((targets[0, indices_batch], targets[1, indices_batch]))
            # set(torch.cat((targets[0][indices_batch], targets[1][indices_batch])).tolist())
            current_data, entity_lst = get_neighbours(node_ids, train_data)
            current_data['entity','target','entity'].edge_label = target_labels[indices_batch]

            # label_lst: list of 1s and 0s indicating where is the target relation in the relation_lst
            label_lst = torch.zeros(current_data['entity', 'target', 'entity'].edge_index.shape[1]*nb_relations,
                                    device=device)
            for i in range(len(current_data['entity', 'target', 'entity'].edge_label)):
                zero_idx = i * nb_relations
                class_num = current_data['entity', 'target', 'entity'].edge_label[i]
                label_lst[zero_idx + class_num] = 1
            # TODO: one_hot-tal megcsinalni a targetet

            labels = torch.tensor(label_lst, dtype=torch.float32, device=device)

            # returns scores of what??
            scores, query_emb_lst = scoring_function_CTP(current_data,
                                                         relation_to_class,
                                                         entity_lst,
                                                         is_train=True,
                                                         _depth=1 if is_simple else None)

            loss = loss_function(scores, labels)

            loss_value = loss.item()
            epoch_loss_values += [loss_value]

            if nb_gradient_accumulation_steps > 1:
                loss = loss / nb_gradient_accumulation_steps

            loss.backward()

            if nb_gradient_accumulation_steps == 1 or global_step % nb_gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.4f}')

            if evaluate_every_batches is not None:
                if global_step % evaluate_every_batches == 0:
                    for test_path in test_paths:
                        # TODO: see if cache deleting needed
                        torch.cuda.empty_cache()
                        evaluate_CTP(data.test_graphs[test_path], relation_to_class, path=test_path)

            torch.cuda.empty_cache()

        if evaluate_every is not None:
            if epoch_no % evaluate_every == 0:
                for test_path in test_paths:
                    evaluate_CTP(graph_data=data.test_graphs[test_path],
                                 relation_to_class=relation_to_class,
                                 path=test_path)

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)

        slope = kernel.slope.item() if isinstance(kernel.slope, Tensor) else kernel.slope
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}\tSlope {slope:.4f}')

    import time
    start = time.time()

    for test_path in test_paths:
        evaluate_CTP(graph_data=data.test_graphs[test_path],
                     relation_to_class=relation_to_class,
                     path=test_path)

    end = time.time()
    logger.info(f'Evaluation took {end - start} seconds.')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    print(' '.join(sys.argv))
    main()
