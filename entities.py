from pathlib import Path

from utils import (
    lines,
    jsonlines_load,
    get_logger,
    file_cached_property,
    cached_property,
    )
from dougu.codecs import LabelEncoder


class Entities():
    def __init__(self, conf, device='cpu'):
        self.conf = conf
        entities_fname = conf.entities_file_tpl.format(top_n=conf.top_n)
        self.entities_file = conf.wikidata_dir / entities_fname
        self.top_n = conf.top_n
        self.device = device
        self.log = get_logger().info

    @cached_property
    def raw(self):
        return list(jsonlines_load(self.entities_file))

    @cached_property
    def ids(self):
        ids_file = Path(str(self.entities_file) + '.ids')
        return list(lines(ids_file))

    @cached_property
    def ids_set(self):
        return set(self.ids)

    @cached_property
    def entity(self):
        return self.ids

    @cached_property
    def labels_en(self):
        fname = self.conf.labels_en_file_tpl.format(top_n=self.top_n)
        labels_en_file = self.conf.wikidata_dir / fname
        return [l.split('\t')[3] for l in lines(labels_en_file)]

    @file_cached_property(fname_tpl='id_enc.{conf_str}.pkl')
    def id_enc(self):
        enc = LabelEncoder(to_torch=True, device=self.device, backend='dict')
        return enc.fit(self.ids)

    @property
    def conf_str(self):
        return f'top{self.top_n}'
