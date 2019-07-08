from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader

from dougu import jsonlines_load, SubclassRegistry, lines, map_assert, flatten
from dougu.decorators import cached_property
from dougu.codecs import LabelEncoder
from synthetic_graph import get_path_sample_file, get_triple_sample_file


class KnowledgeGraphPaths(Dataset, SubclassRegistry):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def load(conf):
        return KnowledgeGraphPaths.get(conf.dataset)(conf)


class SyntheticPaths(KnowledgeGraphPaths):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        paths_file = get_path_sample_file(conf, conf.path_sample_id)
        triples_file = get_triple_sample_file(conf, conf.path_sample_id)
        if not paths_file.exists():
            from synthetic_graph import sample_paths
            sample_paths(conf)
        if not triples_file.exists():
            from synthetic_graph import sample_paths
            triple_conf = deepcopy(conf)
            triple_conf.min_path_len = 3
            triple_conf.max_path_len = 3
            sample_paths(triple_conf)
        print(paths_file)
        self.paths_raw = list(
            jsonlines_load(paths_file, max=conf.n_paths))
        self.triples_raw = list(
            jsonlines_load(triples_file, max=conf.n_paths))
        if conf.min_path_len == conf.max_path_len:
            self.paths = torch.tensor(self.paths_raw)
            if not conf.data_on_cpu:
                self.paths = self.paths.to(device=conf.device)
            assert len(self.paths) == conf.n_paths, len(self.paths)
            self.triples = self.paths
        else:
            if not conf.data_on_cpu:
                def to_tensor(path):
                    return torch.tensor(path).to(device=conf.device)
            else:
                to_tensor = torch.tensor
            paths = torch.cat(list(map(to_tensor, self.paths_raw)))
            seq_len = conf.bptt + 1
            instances = []
            for i in range(seq_len):
                paths_i = paths[i:]
                remainder = divmod(len(paths_i), seq_len)[1]
                paths_i = paths_i[:-remainder or None]
                assert len(paths_i) % seq_len == 0
                instances.append(paths_i.view(-1, seq_len))
            self.paths = torch.cat(instances)
            self.triples = to_tensor(self.triples_raw)

    def __getitem__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)

    @cached_property
    def loader(self):
        return loader(self, self.conf)

    @cached_property
    def loader_trainval(self):
        return loader(
            # self.triples,
            self,
            self.conf,
            max_n_inst=self.conf.max_eval_n_inst,
            shuffle=False)


class Yago3_10(KnowledgeGraphPaths):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.split_names = 'train', 'dev', 'test'
        datadir = conf.datadir / self.__class__.__name__.lower()
        files = [datadir / f'{name}.txt' for name in self.split_names]
        self.splits_raw = list(map(self.load_split_raw, files))
        s, p, o = map(set, zip(*flatten(self.splits_raw)))
        conf.n_nodes = len(s | o)
        conf.n_edge_labels = len(p)
        entity_labels = list(s | o)
        p_labels = list(p)
        self.entity_enc = LabelEncoder(to_torch=True).fit(entity_labels)
        self.p_enc = LabelEncoder(to_torch=True).fit(p_labels)
        print(len(entity_labels), 'entity labels')
        print(len(p_labels), 'p labels')
        for split_name, split_raw in zip(self.split_names, self.splits_raw):
            s, p, o = zip(*split_raw)
            s_enc = self.entity_enc.transform(s)
            p_enc = self.p_enc.transform(p) + len(self.entity_enc.labels)
            o_enc = self.entity_enc.transform(o)
            split = torch.stack([s_enc, p_enc, o_enc], dim=1)
            setattr(self, split_name, split)
            print(split_name, split.shape)
        self.paths = self.train

    def load_split_raw(self, file):
        return list(map_assert(
            str.split, lambda p: len(p) == 3, lines(file)))[:self.conf.n_paths]

    def __getitem__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)

    @cached_property
    def loader(self):
        return loader(self, self.conf)

    @cached_property
    def loader_trainval(self):
        return loader(
            self,
            self.conf,
            max_n_inst=self.conf.max_eval_n_inst,
            shuffle=False)


def loader(dataset, conf, max_n_inst=None, shuffle=True):
    return DataLoader(
        dataset[:max_n_inst],
        batch_size=conf.batch_size,
        shuffle=shuffle)
