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
from kb_final_hetero import BatchNeuralKB
from util_hetero_final_CTP_model import get_neighbours_CTP
from model_hetero_final import BatchHoppy
from gaussian_final import GaussianKernel
from evaluation_heterodata_CTP_model import accuracy

from ctp.reformulators import BaseReformulator
from ctp.reformulators import LinearReformulator
from ctp.reformulators import MemoryReformulator

from typing import Tuple, Dict, Optional

from torch_geometric.data import HeteroData

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=256, precision=4, suppress=True, threshold=sys.maxsize)

torch.set_num_threads(multiprocessing.cpu_count())

class BatcherHetero:
    """The class responsible for creating batches of a HeteroData dataset"""

    def __init__(self,
                 data: HeteroData,
                 batch_size: int,
                 nb_examples: int, # number of targets
                 random_state: Optional[np.random.RandomState]):

        """Initializing some parameters of the BatcherHetero instance.

            Args:
                data (HeteroData): The dataset we want to make batches of.
                batch_size (int): The number of target edges we want to take the connected components of into one batch.
                nb_examples (int): The total number of target edges in the dataset.
                random_state (Optional[np.random.RandomState]): Random state used for the permutation of the
                        target edge indices. If None, no permutation is used.
        """

        self.data = data
        self.batch_size = batch_size
        self.nb_examples = nb_examples
        self.random_state = random_state

        self.curriculum = np.zeros(self.nb_examples, dtype=np.int32)

        if self.random_state is not None:
            self.curriculum = self.random_state.permutation(nb_examples)
        else:
            self.curriculum = np.arange(nb_examples)

        self.batch_indices = make_batches(self.curriculum.shape[0], self.batch_size)
        self.nb_batches = len(self.batch_indices)

        self.targets = self.data['entity', 'target', 'entity'].edge_index
        self.target_labels = self.data['entity', 'target', 'entity'].edge_label

    def get_batch(self, batch_start, batch_end) -> HeteroData:

        """Extracts the batch based on the start and end indices given.

            Args:
                batch_start (int): Starting index giving the position of the first target edge index in the curriculum.
                batch_end (int): Ending index giving the position of the last target edge index in the curriculum.

            Returns:
                HeteroData that is the current batch of the original data.
        """

        indices_batch = self.curriculum[batch_start:batch_end]
        node_ids = torch.cat((self.targets[0, indices_batch], self.targets[1, indices_batch]))
        current_data = get_neighbours_CTP(node_ids, self.data)
        current_data['entity', 'target', 'entity'].edge_label = self.target_labels[indices_batch]
        return current_data


def main():

    """Script for creating, training and running a CTP Model on HeteroData datasets.

        Parameters:
            embedding_size (int): The size of the embedding of each entity and relation in our dataset.
            k_max (int): When proving the body of a rule, we consider the k best substitutions for each entity.
            max_depth (int): This gives how many times to reformulate the goal(s) during training.
            test_max_depth (int): This gives how many times to reformulate the goal(s) during testing.
            hops_str (List[str]): Provides the shape of the reformulations we want to try.
                Possible input format for the strings: string of integer with or without an R at the end.
                The integer value means how many subgoals we create at each reformulation.
                The R means the arguments will be in reversed order in the reformulated subgoals.
                The number of such strings in the list means how many reformulations we want to try
                    (of possibly the same format).
            nb_epochs (int): Number of epochs for training.
            learning_rate (float): The learning rate of training.
            batch_size (int): The number of target edges we want to take the connected components of into one batch
                    during training.
            test_batch_size (int): The number of target edges we want to take the connected components of into one
                    batch during testing.
            optimizer_name (str): The optimizer used for training.
                Choices = 'adagrad', 'adam', 'sgd'.
            seed (int): Seed used for every random during the code for better comparison.
            evaluate_every (Optional[int]): Do evaluation after every 'n'th epoch, 'n' given by this number.
            evaluate_every_batches (Optional[int]): Do evaluation after every 'n'th batch, 'n' given by this number.
            tnorm_name (str): This gives the method of how to update scores while reformulating the current
                goal into subgoals.
                'min': The function updates scores by taking the minimum of all embedding similarity scores so far.
                'prod': The function updates scores by taking the product of the embedding similarity scores so far.
                'mean': The function updates scores by taking the mean of all embedding similarity scores so far.
            reformulator_name (str): Gives the type of reformulation function we want to use and learn.
                Currently only available: 'linear'.
                Possible other choices: 'static', 'linear', 'attentive', 'memory', 'ntp'.
            nb_gradient_accumulation_steps (int) : Gives the number of gradient accumulation steps.
            slope (float): Gives the slope for the Gaussian Kernel.
            init_size (float): Provides the initial size of the weights in the relation embeddings.
            init_type (str): Provides the type of initial distribution of the relation embedding weights.
                Choices: 'random', 'uniform'.
            ref_init_type (str): Provides the type of the initial distribution of weights for the reformualtion.
                Choices: 'random', 'uniform'.
            load_path (str): Path to load the model state into.
            save_path (str): Path to save the model state into.
    """

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

    embedding_size = 20
    k_max = 5
    max_depth = 2
    test_max_depth = 2
    hops_str = ['2', '2', '2']


    # training params

    nb_epochs = 50
    learning_rate = 0.1
    batch_size = 4
    test_batch_size = 2

    optimizer_name = 'adagrad'
    seed = 1

    evaluate_every = None
    evaluate_every_batches = None

    tnorm_name = 'min'
    reformulator_name = 'linear'

    nb_gradient_accumulation_steps = 1
    slope = 1.0
    init_size = 1.0

    init_type = 'random'
    ref_init_type = 'random'

    load_path = None
    save_path = None

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # loading the data into HeteroData format
    data = DataParserCTP(train_path=train_path, test_paths=test_paths)
    entity_lst, relation_lst = data.entity_lst, data.relation_lst

    nb_entities = len(entity_lst)
    nb_relations = len(relation_lst)

    kernel = GaussianKernel(slope=slope)

    # the model that lets you look up in the KB
    model = BatchNeuralKB(kernel=kernel, scoring_type='concat').to(device)
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

    class CTPDecoder(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.depth = depth
            self.relation_embeddings = nn.Embedding(nb_relations, embedding_size, sparse=True).to(device)
            self.relation_embeddings.requires_grad = False
            if init_type in {'uniform'}:
                nn.init.uniform_(self.relation_embeddings.weight, -1.0, 1.0)
            self.relation_embeddings.weight.data *= init_size

        def forward(self, x_dict, edge_index_dict, edge_label_index, is_train):
            # Encoding targets

            # [nb_targets*R,E]
            rel_idx = torch.tile(torch.arange(nb_relations), (edge_label_index.shape[1], 1)).flatten()
            rel = self.relation_embeddings(rel_idx)
            arg1_idx = torch.tile(edge_label_index[0, :], (nb_relations, 1)).T.flatten()
            arg1 = x_dict['entity'][arg1_idx]
            arg2_idx = torch.tile(edge_label_index[1, :], (nb_relations, 1)).T.flatten()
            arg2 = x_dict['entity'][arg2_idx]

            # Encoding facts

            # F: total number of facts
            # necessary that the target edge type is the last edge type
            # [F,E]
            facts_arg1 = torch.cat([x_dict['entity'][edge_index[0, :]] for edge_index in list(edge_index_dict.values())[:-1]])
            facts_arg2 = torch.cat([x_dict['entity'][edge_index[1, :]] for edge_index in list(edge_index_dict.values())[:-1]])

            rel_index_lst = []
            for key, edge_index in edge_index_dict.items():
                if key[1] != 'target':  # we don't want to include the target edges in the facts
                    rel_index_lst.extend([relation_to_class[key[1]]] * edge_index.shape[1])
            rel_index = torch.tensor(rel_index_lst, dtype=torch.long, device=device)

            # [F,E]
            facts_rel = self.relation_embeddings(rel_index)

            # nb_targets*R = B
            batch_size = rel.shape[0]

            # N - the number of different entities in the facts
            entity_size = x_dict['entity'].shape[0]

            # [B, N, E]
            # repeat the same entity embeddings for each instance in batch (== for each relation)
            # (1 batch will be all possible relation substitutions for the target relation)
            # --> embeddings_lst will become [batch_size,N,E], where batch is number of instances in batch
            # embeddings = embeddings.view(1, entity_size, -1).repeat(batch_size, 1, 1)
            embeddings = x_dict['entity'].view(1, entity_size, -1).repeat(batch_size, 1, 1)

            # [3,F,E]
            facts = [facts_rel, facts_arg1, facts_arg2]

            max_depth_ = hoppy.depth
            if not is_train and test_max_depth is not None:
                hoppy.depth = test_max_depth

            if self.depth is not None:
                hoppy.depth = self.depth

            # [nb_targets*R] = [B]
            scores = hoppy.prove(rel, arg1, arg2, facts, embeddings, hoppy.depth)

            if not is_train and test_max_depth is not None:
                hoppy.depth = max_depth_

            if self.depth is not None:
                hoppy.depth = max_depth_

            return scores, [rel, arg1, arg2]

    class CTPModel(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True).to(device)
            nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)
            self.embeddings.requires_grad = False
            self.decoder = CTPDecoder(depth).to(device)

        def forward(self, x_dict, edge_index_dict, edge_label_index, is_train=False):
            x_dict = {'entity': self.embeddings(x_dict['entity'])}
            return self.decoder(x_dict, edge_index_dict, edge_label_index, is_train)

    def evaluate_CTP(graph_data: HeteroData,
                     ctp_model: nn.Module,
                     path: str) -> float:
        res = accuracy(ctp_model=ctp_model,
                       graph_data=graph_data,
                       relation_lst=relation_lst,
                       batch_size=test_batch_size)
        logger.info(f'Test Accuracy on {path}: {res:.6f}')
        return res

    loss_function = nn.BCELoss()

    params_lst = list(hoppy.parameters())

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

    ctp_model = CTPModel(None).to(device)

    for epoch_no in range(1, nb_epochs + 1):

        train_data, is_simple = data.train_graph, False

        targets = train_data['entity', 'target', 'entity'].edge_index

        batcher = BatcherHetero(data=train_data, batch_size=batch_size, nb_examples=targets.shape[1], random_state=random_state)

        nb_batches = batcher.nb_batches
        epoch_loss_values = []

        for batch_no, (batch_start,batch_end) in enumerate(batcher.batch_indices, start=1):
            global_step += 1

            current_data = batcher.get_batch(batch_start,batch_end)


            # label_lst: list of 1s and 0s indicating where is the target relation in the relation_lst
            labels = torch.zeros(current_data['entity', 'target', 'entity'].edge_index.shape[1]*nb_relations,
                                    device=device)
            for i in range(len(current_data['entity', 'target', 'entity'].edge_label)):
                zero_idx = i * nb_relations
                class_num = current_data['entity', 'target', 'entity'].edge_label[i]
                labels[zero_idx + class_num] = 1


            # returns scores for each edge type substituted for each target edge
            scores, query_emb_lst = ctp_model(current_data.x_dict,
                                              current_data.edge_index_dict,
                                              current_data['entity', 'target', 'entity'].edge_index,
                                              is_train=True)

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
                        torch.cuda.empty_cache()
                        evaluate_CTP(data.test_graphs[test_path],  ctp_model, path=test_path)
                    evaluate_CTP(data.train_graph, ctp_model, path=train_path)

            torch.cuda.empty_cache()

        if evaluate_every is not None:
            if epoch_no % evaluate_every == 0:
                torch.cuda.empty_cache()
                for test_path in test_paths:
                    evaluate_CTP(graph_data=data.test_graphs[test_path],
                                 ctp_model=ctp_model,
                                 path=test_path)
                evaluate_CTP(data.train_graph, ctp_model, path=train_path)

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)

        slope = kernel.slope.item() if isinstance(kernel.slope, Tensor) else kernel.slope
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}\tSlope {slope:.4f}')

    import time
    start = time.time()

    for test_path in test_paths:
        evaluate_CTP(graph_data=data.test_graphs[test_path],
                     ctp_model=ctp_model,
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
