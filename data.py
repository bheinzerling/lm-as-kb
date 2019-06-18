import torch
from torch.utils.data import Dataset, DataLoader

from dougu import jsonlines_load, SubclassRegistry
from dougu.decorators import cached_property
from synthetic_graph import get_path_sample_file


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
        if not paths_file.exists():
            from synthetic_graph import sample_paths
            sample_paths(conf)
        self.paths_raw = list(
            jsonlines_load(paths_file, max=conf.n_paths))
        self.tensor = torch.tensor(self.paths_raw)
        assert len(self.tensor) == conf.n_paths
        if not conf.data_on_cpu:
            self.tensor = self.tensor.to(device=conf.device)

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)

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
