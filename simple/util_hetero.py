import copy

from typing import List, Tuple, Dict, Optional

import torch
from torch_geometric.data import HeteroData


# TODO: find bugs
# gives back the graph containing only the node_ids that are connected to the given node ids
def get_neighbours(node_ids: set[int],
                   graph_data: HeteroData,
                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> (HeteroData, List[int]):

    res_graph = HeteroData()

    # getting all nodes that are connected to the given nodes
    prev_size = -1
    new_size = len(node_ids)
    while new_size > prev_size:
        for edge in graph_data.edge_types:
            # print(f"edge: {edge}")
            # print(f"graph_data[edge]: {graph_data[edge]}")
            for s, o in zip(graph_data[edge].edge_index[0], graph_data[edge].edge_index[1]):
                if s in node_ids:
                    if o not in node_ids:
                        node_ids.add(o)
                elif o in node_ids:
                    node_ids.add(s)
        prev_size = new_size
        new_size = len(node_ids)

    # getting rid of nodes and edges that doesn't contain the nodes collected above
    # assume: nodes are still ids at this point and not embeddings
    for node in graph_data.node_types:
        new_nodes = []
        for n in graph_data[node].x:
            if int(n) in node_ids:
                new_nodes.append(int(n))
        res_graph[node].x = torch.tensor(new_nodes, dtype=torch.long, device=device)

    for edge in graph_data.edge_types:
        new_edges = [[], []]
        new_target_labels = []
        idx_for_target_label = 0
        for s, o in zip(graph_data[edge].edge_index[0], graph_data[edge].edge_index[1]):
            if int(s) in node_ids:  # then o is in it as well
                new_edges[0].append(int(s))
                new_edges[1].append(int(o))
                if edge[1] == 'target':
                    new_target_labels.append(graph_data[edge].edge_label[idx_for_target_label])
            idx_for_target_label += 1
        res_graph[edge].edge_index = torch.tensor(new_edges, dtype=torch.long, device=device)
        if edge[1] == 'target':
            res_graph[edge].edge_label = torch.tensor(new_target_labels, dtype=torch.long, device=device)

    entity_list = sorted(node_ids)

    print(res_graph)

    return res_graph, entity_list
