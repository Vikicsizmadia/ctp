import csv

import torch
import yaml
import json
from os.path import join, dirname, abspath

from collections import OrderedDict

from typing import List, Tuple, Any, Optional, Dict
from torch_geometric.data import HeteroData


# parses the data given from a csv specified by the path inputs
class DataParser:
    def __init__(self,
                 train_path: str = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp',
                                        'data_089907f8', '1.2,1.3_train.csv'),
                 test_paths: Optional[List[str]] = None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        self.device = device

        with open(join(dirname(dirname(abspath(__file__))), "data", "clutrr-emnlp", "relations_store.yaml"),
                  'r') as f:
            rs = yaml.safe_load(f)
        self.relation_to_predicate = {r['rel']: k for k, v in rs.items()
                                      for _, r in v.items() if k != 'no-relation'}
        self.relation_lst = sorted({r for r in self.relation_to_predicate.keys()})

        allowed_edges = sorted({'father', 'son', 'wife', 'husband', 'uncle', 'grandfather', 'grandmother', 'daughter'})
        # self.relation_lst = allowed_edges

        self.edge_types_to_class = {}  # dictionary to connect relation types to class ids
        class_num = 0
        for rel in self.relation_lst:
            self.edge_types_to_class[rel] = class_num
            class_num += 1

        (self._train_graph, self._train_nodes_ids, new_idx, edge_types) = DataParser.parse(train_path, 0,
                                                                                           self.device,
                                                                                           self.edge_types_to_class,
                                                                                           allowed_edges)
                                                                                           # self.relation_lst)
        entity_set = set(self.train_nodes_ids.keys())

        self._test_graphs = OrderedDict()
        self._test_nodes_ids = OrderedDict()
        for test_path in (test_paths if test_paths is not None else []):
            (self._test_graphs[test_path], self._test_nodes_ids[test_path], new_idx, test_edge_types) = \
                DataParser.parse(test_path, new_idx, self.device, self.edge_types_to_class, allowed_edges) # edge_types)
            entity_set |= set(self.test_nodes_ids[test_path].keys())

        self.entity_lst = sorted(entity_set)

    @property
    def train_graph(self) -> HeteroData:
        return self._train_graph

    @property
    def train_nodes_ids(self) -> Dict[str, int]:
        return self._train_nodes_ids

    @property
    def test_graphs(self) -> OrderedDict[str, HeteroData]:
        return self._test_graphs

    @property
    def test_nodes_ids(self) -> OrderedDict[str, Dict[str, int]]:
        return self._test_nodes_ids

    @staticmethod
    def _to_obj(s: str) -> Any:
        return json.loads(s.replace(")", "]").replace("(", "[").replace("'", "\""))

    # process the csv file at the given path
    # returning the graph, the target edges, the node names corresponding to their ids in the graph
    @staticmethod
    def parse(path: str, idx: int, device, edge_types_to_class, allowed_edges) -> (HeteroData, Dict[str, int], int):
        original_idx = idx
        data = HeteroData()
        with open(path, newline='') as f:
            reader = csv.reader(f)
            # idx: to add to names to make them unique for each mini graph
            name_to_id = {}  # dictionary to collect unique entities and their unique ids
            edges = {}  # dictionary for collecting edges of different types (edge_type : [[subject_nodes],[object_nodes]])
            target_edges = [[], []]  # collecting target edges ([[subject_nodes],[object_nodes]])
            target_labels = [] # collecting corresponding target labels (["father","uncle",...])
            #edge_types_to_class = {}  # dictionary to collect relation types to corresponding class ids
            for row in reader:
                _id, _, _, query, _, target, _, _, _, _, _, story_edges, edge_types, _, genders, _, tmp, _ = row
                if len(_id) > 0:
                    # add_to_id: this many entities we have so far, so subsequent entities are numbered from there
                    add_to_id = len(name_to_id)
                    name_to_id |= {name.split(':')[0] + str(idx): (i + add_to_id) for i, name in
                                   enumerate(genders.split(','))}
                    _story, _edge, _query = DataParser._to_obj(story_edges), DataParser._to_obj(edge_types), DataParser._to_obj(query)
                    for (s_id, o_id), p in zip(_story, _edge):
                        if p not in edges.keys():
                            edges[p] = [[], []]
                        edges[p][0].append(s_id + add_to_id)
                        edges[p][1].append(o_id + add_to_id)
                    target_edges[0].append(name_to_id[_query[0] + str(idx)])
                    target_edges[1].append(name_to_id[_query[1] + str(idx)])
                    target_labels.append(edge_types_to_class[target])

                idx += 1

            # just the indices
            data['entity'].x = torch.arange(original_idx, original_idx+len(name_to_id), dtype=torch.long, device=device)
            #print(edges.keys())
            #for rel_type in edges.keys():
            #    if rel_type in allowed_edges:
            for rel_type in allowed_edges:
                data['entity', rel_type, 'entity'].edge_index = torch.tensor(edges[rel_type],
                                                                             dtype=torch.long, device=device)
            data['entity', 'target', 'entity'].edge_index = torch.tensor(target_edges, dtype=torch.long, device=device)
            data['entity', 'target', 'entity'].edge_label = torch.tensor(target_labels, dtype=torch.long, device=device)
        return data, name_to_id, idx, edges.keys()
