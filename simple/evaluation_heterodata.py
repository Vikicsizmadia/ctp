# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import Tensor

from ctp.clutrr.base import Instance
from ctp.util import make_batches

from typing import Callable, List, Optional, Tuple, Any, Dict

from torch_geometric.data import HeteroData
from simple import get_neighbours


def accuracy(scoring_function: Callable[[HeteroData, Dict[str, int], List[int]], Tuple[Tensor, Any]],
             graph_data: HeteroData,  # instances: List[Instance],
             relation_to_class: Dict[str, int],
             relation_lst: List[str],
             # sample_size: Optional[int] = None,
             entity_lst: List[int],
             batch_size: Optional[int] = None,
             # relation_to_predicate: Optional[Dict[str, str]] = None,
             is_debug: bool = False) -> float:

    # if sample_size is not None:
    #    instances = instances[:sample_size]
    targets = graph_data['entity', 'target', 'entity'].edge_index
    nb_instances = targets.shape[1]

    # nb_instances = len(instances)

    batches = [(None, None)]
    if batch_size is not None:
        batches = make_batches(nb_instances, batch_size)

    nb_relations = len(relation_lst)

    is_correct_lst = []

    # for batch_start, batch_end in batches:
    #    batch = instances[batch_start:batch_end]
    #    batch_size = len(batch)

    for batch_start, batch_end in batches:

        # getting current batch from the training set
        indices_batch = np.arange(batch_start, batch_end)
        #indices_batch = batcher.get_batch(batch_start, batch_end)
        node_ids = set(torch.cat((targets[0][indices_batch], targets[1][indices_batch])).tolist())
        current_data, entity_lst = get_neighbours(node_ids, graph_data)

        batch_size = batch_end-batch_start

        with torch.no_grad():
            scores, _ = scoring_function(current_data, relation_to_class, entity_lst)
            scores = scores.view(batch_size, nb_relations)
            scores_np = scores.cpu().numpy()

        predicted = np.argmax(scores_np, axis=1)

        # np does not support CUDA, so need to put tensor to cpu
        true = np.array(current_data['entity', 'target', 'entity'].edge_label.cpu())
        # ([relation_lst.index(i.target[1]) for i in batch], dtype=predicted.dtype)


        # for i, (a, b) in enumerate(zip(predicted.tolist(), true.tolist())):
        #    if a != b:
        #        if is_debug is True:
        #            print(batch[i])
        #            rel_score_pairs = [(relation_lst[j], scores_np[i, j]) for j in range(len(relation_lst))]
        #            print(rel_score_pairs)

        is_correct_lst += (predicted == true).tolist()

    return np.mean(is_correct_lst).item() * 100.0
