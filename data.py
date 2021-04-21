import torch
from torch.utils.data import DataLoader

from utils import (
    jsonlines_load,
    get_logger,
    torch_cached_property,
    cached_property,
    SubclassRegistry,
    )

from entities import Entities


class TensorDictDataset():
    """Like a TensorDataset, but instead of a tuple of tensors,
    __getitem__ returns a dict of tensors, which makes code using
    this dataset more readable.
    """
    def __init__(self, tensor_dict):
        self.tensors = tensor_dict

    def __getitem__(self, index):
        return {k: tensor[index] for k, tensor in self.tensors.items()}

    def __len__(self):
        return len(next(iter(self.tensors.values())))


class Data(SubclassRegistry):
    """Baseclass for data.
    """

    def __init__(self, conf):
        self.log = get_logger().info
        self.conf = conf
        self.max_seq_len = conf.max_seq_len

    @cached_property
    def entities(self):
        return Entities(self.conf)

    @property
    def conf_str(self):
        """A string that uniquely describes the configuration of this data. Used
        for automatic caching.
        """
        fields = ['top_n', 'max_seq_len']
        if self.conf.max_train_inst:
            fields.append('max_train_inst')
        if self.conf.max_eval_inst:
            fields.append('max_eval_inst')
        if self.conf.max_test_inst:
            fields.append('max_test_inst')
        if self.conf.paraphrase_id is not None:
            fields.append('paraphrase_id')
        return '.'.join([
            field + str(getattr(self.conf, field)) for field in fields])

    @cached_property
    def raw(self):
        """Raw data as read from a file on disk, which has not been
        tensorized yet.
        """
        return self.load_raw_data()

    def get_max_inst(self, split_name):
        """The maximum number of instances to be included in the
        specified split.
        """
        test_name = 'test' if self.conf.max_test_inst else 'eval'
        split_name = {
            'train': 'train',
            'dev': 'eval',
            'test': test_name}[split_name]
        return getattr(self.conf, f'max_{split_name}_inst')

    def load_raw_data(self):
        """Load raw data for all splits from disk."""
        return {
            split_name:
                self.load_raw_split(split_name)[:self.get_max_inst(split_name)]
            for split_name in self.split_names}

    @cached_property
    def train_sampler(self):
        """The sampler for the training split. Returns a DistributedSampler
        if we're doing distributed training, defaults to PyTorch's default
        sampler otherwise.
        """
        if self.conf.distributed:
            return torch.utils.data.distributed.DistributedSampler(
                self.train,
                num_replicas=torch.distributed.get_world_size(),
                rank=self.conf.local_rank,
                shuffle=True)
        return None

    def has_train_data(self):
        """Use to check if this is a test-only dataset.
        """
        return len(next(iter(self.tensors['train'].values()))) > 0

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.conf.tokenizer)

    @torch_cached_property(
        fname_tpl='tensors.{conf_str}.pth', map_location='cpu')
    def tensors(self):
        """The data in tensorized form, ready to be fed directly to the model.
        This method is backed by a file cache, that is, the raw data will be
        tensorized one (possibly very slow) and cached. Subsequent runs will
        then load tensors from this cache, which is usually much faster.
        """
        return self.tensorize()

    def tensorize(self):
        """Turn raw data from each split into PyTorch tensors.
        """
        return {
            split_name: self.tensorize_split(split)
            for split_name, split in self.raw.items()}

    def log_size(self):
        """Log size of each split.
        """
        for split_name in self.split_names:
            split = getattr(self, split_name)
            if split is not None:
                msg = f'{len(split)} {split_name} instances'
                loader_name = split_name + '_loader'
                loader = getattr(self, loader_name)
                if loader is not None:
                    msg += f' | {len(loader)} batches'
                self.log(msg)


class EvalOnTrain():
    """A Dataset without test set.
    """
    @property
    def split_names(self):
        return ['train', 'dev']

    @cached_property
    def train(self):
        return TensorDictDataset(self.tensors['train'])

    @property
    def dev(self):
        return TensorDictDataset(self.tensors['dev'])

    @property
    def train_loader(self):
        return DataLoader(
            self.train,
            batch_size=self.conf.batch_size,
            pin_memory=False,
            num_workers=0,
            sampler=self.train_sampler)

    @property
    def dev_loader(self):
        return self.eval_loader(self.dev)

    def eval_loader(self, tensorized_data):
        return DataLoader(
            tensorized_data,
            self.conf.eval_batch_size,
            shuffle=False,
            )


class TrainDevTest():
    """A datase with typical train/dev/test splits.
    """
    @property
    def split_names(self):
        return ['train', 'dev', 'test']

    @property
    def train(self):
        return TensorDictDataset(self.tensors['train'])

    @property
    def dev(self):
        return TensorDictDataset(self.tensors['dev'])

    @property
    def test(self):
        return TensorDictDataset(self.tensors['test'])

    @property
    def train_loader(self):
        return DataLoader(
            self.train,
            self.conf.batch_size,
            sampler=self.train_sampler)

    @property
    def dev_loader(self):
        return self.eval_loader(self.dev)

    @property
    def test_loader(self):
        return self.eval_loader(self.test)

    def eval_loader(self, tensorized_data):
        return DataLoader(
            tensorized_data,
            self.conf.eval_batch_size,
            shuffle=False)


class RelationStatements(Data):
    """A dataset contain relation statements such as "Barack Obama
    was born in Hawaii". The object entity ("Hawaii") will be masked
    to create training instances like "Barack Obama was born in [MASK]".
    """
    is_pretokenized = False

    def tensorize_split(self, raw_data):
        """Tokenize and encode relation statements, also add tensors
        contiaining entity ids and their positions in the statements.
        """
        instances = self.raw_data_to_instances(raw_data)
        if not instances['contexts']:
            return {'dummy': []}
        tensors = self.tokenizer.batch_encode_plus(
            instances['contexts'],
            max_length=self.conf.max_seq_len,
            padding='max_length',
            return_tensors='pt',
            is_split_into_words=self.is_pretokenized)
        # the object returned by batch_encode_plus cannot be serialized
        # with torch.save, so we convert it into a dict, which can
        tensors = dict(tensors)
        tensors['entity_ids'] = instances['entity_ids']
        entity_mask = tensors['input_ids'] == self.tokenizer.mask_token_id
        if entity_mask.sum() != len(entity_mask):
            # in a few very rare cases, the first entity's name is so long that
            # the second entitity is truncated. Arbitrarily turn the last token
            # into an entity mention in these cases to ensure that each
            # trunacated statement also has exactly one object entity.
            missing = entity_mask.sum(1) != 1
            entity_mask[missing, -2] = 1
        tensors['entity_mask'] = entity_mask
        return tensors

    def untensorize(self, tensors, pred):
        """The reverse of tensorize: Decode tensors into text.
        """
        attn_masks = tensors['attention_mask'].bool()
        for input_ids, attn_mask, entity_target_id, entity_pred_id in zip(
                tensors['input_ids'], attn_masks, tensors['entity_ids'], pred):
            context = self.tokenizer.decode(input_ids.masked_select(attn_mask))
            context = context.replace(
                self.tokenizer.mask_token, ' ' + self.tokenizer.mask_token)
            entity_target = self.entities.labels_en[entity_target_id]
            entity_pred = self.entities.labels_en[entity_pred_id]
            yield {
                'context': context,
                'entity_target': entity_target,
                'entity_pred': entity_pred}

    def to_context_target_pred(self, batch, output):
        """Convenience function similar to untensorize.
        """
        pred = output['entity_pred'][:, 0]
        instances = self.untensorize(batch, pred)
        for i in instances:
            yield i['context'], i['entity_target'], i['entity_pred']

    def raw_data_to_instances(self, raw_data):
        raise NotImplementedError()

    def mask_entity(self, sent_tokens, entity_start_idx, entity_end_idx):
        return (
            sent_tokens[:entity_start_idx] +
            [self.tokenizer.mask_token] +
            sent_tokens[entity_end_idx:])


class WikidataRelationStatements_Base(RelationStatements):
    """Baseclass for statements derived from Wikidata relations. The only
    puprose of this class is to encapsulate two methods for reading Wikidata
    files.
    """
    target_arg = 'o'  # use relation objects as targets

    def load_raw_split(self, split_name):
        fname = self.filename_for_split(split_name)
        split_file = self.conf.datadir / self.dir_name / fname
        if split_name in {'dev', 'test'}:
            if split_name == 'test' and self.conf.max_test_inst:
                n_inst = self.conf.max_test_inst
            else:
                n_inst = self.conf.max_eval_inst
        else:
            n_inst = self.conf.n_facts
        instances_raw = list(jsonlines_load(split_file, max=n_inst))
        assert n_inst == len(instances_raw), len(instances_raw)
        return instances_raw

    def raw_data_to_instances(self, raw_data):
        start_idx_key = self.target_arg + '_start'
        end_idx_key = self.target_arg + '_end'
        target_id_key = self.target_arg + 'id'
        contexts = [
            ' '.join(self.mask_entity(
                inst['sent'],
                entity_start_idx=inst[start_idx_key],
                entity_end_idx=inst[end_idx_key]))
            for inst in raw_data]
        target_ids_raw = [inst[target_id_key] for inst in raw_data]
        target_ids = self.entities.id_enc.transform(target_ids_raw)
        return {
            'contexts': contexts,
            'entity_ids': target_ids}


class WikidataRelationStatements(WikidataRelationStatements_Base, EvalOnTrain):
    """Statements derived from Wikidata relations, involving the top n entities
    in Wikidata, where n = 1 million or n = 6 million.
    """
    @property
    def dir_name(self):
        return 'wikidata_relation_statements'

    def filename_for_split(self, split_name):
        return f'relation_statements.en.top{self.conf.top_n}.jl'


class WikidataParaphrases(WikidataRelationStatements_Base, TrainDevTest):
    """Statements derived from Wikidata relations, with predefined splits for
    paraphrases and finetuning.
    """
    @property
    def dir_name(self):
        return 'wikidata_paraphrases'

    def filename_for_split(self, split_name):
        i = self.conf.paraphrase_id
        if self.conf.paraphrase_mode == 'train':
            split_name = ''
        else:
            assert self.conf.paraphrase_mode == 'finetune'
            if split_name in {'train', 'dev'}:
                split_name = 'finetune.'
            else:
                assert split_name == 'test'
                split_name = 'eval.'
        rel = self.relation_name
        return f'wd-paraphrase.{rel}.{i}{split_name}.top{self.conf.top_n}.jl'


class WikidataParaphrases_born_in(WikidataParaphrases):
    relation_name = 'born_in'
