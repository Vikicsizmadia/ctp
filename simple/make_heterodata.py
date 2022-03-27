import csv

import torch
import yaml
import json
from os.path import join, dirname, abspath

from collections import OrderedDict

from typing import List, Tuple, Any, Optional, Dict
from torch_geometric.data import HeteroData

Fact = Tuple[str, str, str]
Story = List[Fact]


# parses the data given from a csv specified by the path inputs
class Data:
    def __init__(self,
                 train_path: str = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp',
                                        'data_089907f8', '1.2,1.3_train.csv'),
                 # 'data/clutrr-emnlp/data_089907f8/1.2,1.3_train.csv',
                 test_paths: Optional[List[str]] = None):

        with open(join(dirname(dirname(dirname(abspath(__file__)))), "data", "clutrr-emnlp", "relations_store.yaml"),
                  'r') as f:
            rs = yaml.safe_load(f)

        self.relation_to_predicate = {r['rel']: k for k, v in rs.items()
                                      for _, r in v.items() if k != 'no-relation'}

        self.relation_lst = sorted({r for r in self.relation_to_predicate.keys()})
        self.predicate_lst = sorted({p for p in self.relation_to_predicate.values()})

        self.predicate_to_relations = {p: [r for r in self.relation_lst if p == self.relation_to_predicate[r]]
                                       for p in self.predicate_lst}

        self._train_instances = Data.parse(train_path)
        entity_set = {s for i in self.train for (s, _, _) in i.story} | {o for i in self.train for (_, _, o) in i.story}

        self._test_instances = OrderedDict()
        for test_path in (test_paths if test_paths is not None else []):
            i_lst = self._test_instances[test_path] = Data.parse(test_path)
            entity_set |= {s for i in i_lst for (s, _, _) in i.story} | {o for i in i_lst for (_, _, o) in i.story}

        self.entity_lst = sorted(entity_set)

        for instance in self.train:
            for s, r, o in instance.story:
                assert s in self.entity_lst and o in self.entity_lst
                assert r in self.relation_lst

    @property
    def train(self) -> List[Instance]:
        return self._train_instances

    @property
    def test(self) -> Dict[str, List[Instance]]:
        return self._test_instances

    @staticmethod
    def _to_obj(s: str) -> Any:
        return json.loads(s.replace(")", "]").replace("(", "[").replace("'", "\""))

    # extracting the graph data and the target edges from the csv file
    @staticmethod
    def parse(path: str) -> (HeteroData, Dict[str, List[List[int]]]):
        data = HeteroData()
        with open(path, newline='') as f:
            reader = csv.reader(f)
            idx = 0  # to add to names to make them unique for each mini graph
            name_to_id = {}  # dictionary to collect unique entities and their unique ids
            edges = {}  # dictionary for collecting edges of different types (edge_type : [[subject_nodes],[object_nodes]])
            target_edges = {}  # dictionary for collecting target edges (edge_type : [[subject_nodes],[object_nodes]])
            for row in reader:
                _id, _, _, query, _, target, _, _, _, _, _, story_edges, edge_types, _, genders, _, tmp, _ = row
                if len(_id) > 0:
                    # nb_nodes = int(tmp[tmp.rfind(":") + 2:-1]) + 1
                    add_to_id = len(
                        name_to_id)  # this many entities we have so far, so subsequent entities are numbered from there
                    name_to_id |= {name.split(':')[0] + str(idx): (i + add_to_id) for i, name in
                                   enumerate(genders.split(','))}
                    _story, _edge, _query = Data._to_obj(story_edges), Data._to_obj(edge_types), Data._to_obj(query)
                    for (s_id, o_id), p in zip(_story, _edge):
                        if p not in edges.keys():
                            edges[p] = [[], []]
                        edges[p][0].append(s_id + add_to_id)
                        edges[p][1].append(o_id + add_to_id)
                    if target not in target_edges.keys():
                        target_edges[target] = [[], []]
                    target_edges[target][0].append(name_to_id[_query[0]])
                    target_edges[target][1].append(name_to_id[_query[1]])

                idx += 1

            data['entity'].x = torch.arange(0, len(name_to_id)).view(len(name_to_id), 1)  # just the indices
            for rel_type in edges.keys():
                data['entity', rel_type, 'entity'].edge_index = torch.Tensor(edges[rel_type])
        return data, target_edges
