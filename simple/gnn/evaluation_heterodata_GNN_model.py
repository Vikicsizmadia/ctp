# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import Tensor
from typing import Callable, List, Dict

from torch_geometric.data import HeteroData


def accuracyGNN(gnn_model: Callable[[Tensor, Dict, Tensor], Tensor],
                graph_data: HeteroData,
                relation_lst: List[str]) -> float:

    # targets = graph_data['entity', 'target', 'entity'].edge_index
    # target_labels = graph_data['entity', 'target', 'entity'].edge_label
    # nb_instances = targets.shape[1]

    #batches = [(None, None)]
    #if batch_size is not None:
    #    batches = make_batches(nb_instances, batch_size)

    nb_relations = len(relation_lst)

    is_correct_lst = []

    #for batch_start, batch_end in batches:

        # getting current batch from the training set
    #    indices_batch = np.arange(batch_start, batch_end)
    #    node_ids = torch.cat((targets[0][indices_batch], targets[1][indices_batch]))
    #    current_data, entity_lst = get_neighbours(node_ids, graph_data)
    #    current_data['entity', 'target', 'entity'].edge_label = target_labels[indices_batch]

    #    batch_size = batch_end-batch_start

    with torch.no_grad():
        scores = gnn_model(graph_data.x_dict,
                           graph_data.edge_index_dict,
                           graph_data['entity', 'target', 'entity'].edge_index)
        scores = scores.view(-1, nb_relations)
        scores_np = scores.cpu().numpy()

    predicted = np.argmax(scores_np, axis=1)

    # np does not support CUDA, so need to put tensor to cpu
    true = np.array(graph_data['entity', 'target', 'entity'].edge_label.cpu())

    is_correct_lst += (predicted == true).tolist()

    return np.mean(is_correct_lst).item() * 100.0
