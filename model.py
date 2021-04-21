import torch
from torch import nn

from utils import SubclassRegistry


def custom_transformer(model_name, *, n_hidden, n_layers, automodel_cls=None):
    if automodel_cls is None:
        from transformers import AutoModel as automodel_cls
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    ratio = config.intermediate_size // n_hidden
    config.hidden_size = n_hidden
    config.intermediate_size = ratio * n_hidden
    config.num_hidden_layers = n_layers
    return automodel_cls.from_config(config)


class KBMemory(nn.Module, SubclassRegistry):
    def __init__(self, conf, vocab_size, padding_idx):
        super().__init__()
        self.conf = conf
        self.setup_encoder(vocab_size, padding_idx)
        self.setup_entity_head()

    def setup_entity_head(self):
        cls = globals()['EntityHead_' + self.conf.entity_repr]
        self.entity_head = cls(self.conf, self.entity_repr_dim)

    def forward(self, batch):
        out = self.encode(batch)
        entity_repr = out[batch['entity_mask']]
        return self.entity_head(entity_repr, target=batch.get('entity_ids'))


class KBMemory_Transformer(KBMemory):
    def setup_encoder(self, vocab_size, padding_idx):
        r = '-randinit'
        if self.conf.transformer_model.endswith(r):
            model_name = self.conf.transformer_model[:-len(r)]
            model = custom_transformer(
                model_name,
                n_hidden=self.conf.n_hidden,
                n_layers=self.conf.n_layers)
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(self.conf.transformer_model)
        self.encoder = model
        self.conf.n_hidden = self.encoder.config.hidden_size
        self.conf.n_layers = self.encoder.config.num_hidden_layers

    @property
    def entity_repr_dim(self):
        return self.encoder.config.hidden_size

    def encode(self, batch):
        return self.encoder(batch['input_ids'], batch['attention_mask'])[0]


class KBMemory_LSTM(KBMemory):
    def setup_encoder(self, vocab_size, padding_idx):
        c = self.conf
        self.emb = nn.Embedding(
            vocab_size, c.rnn_emb_dim, padding_idx=padding_idx)
        self.encoder = getattr(nn, c.rnn_type)(
            input_size=c.rnn_emb_dim,
            hidden_size=c.n_hidden,
            dropout=c.rnn_dropout,
            num_layers=c.n_layers,
            bidirectional=True,
            batch_first=True)
        self.n_hidden = c.n_hidden
        self.n_layers = c.n_layers
        self.repr_dim = 2 * c.n_hidden

    @property
    def entity_repr_dim(self):
        return self.repr_dim

    def encode(self, batch):
        emb = self.emb(batch['input_ids'])
        return self.encoder(emb)[0]


class EntityHead_symbol(nn.Module):
    def __init__(self, conf, entity_repr_dim):
        super().__init__()
        self.repr_to_softmax = nn.Linear(entity_repr_dim, conf.softmax_size)
        self.proj = nn.Linear(conf.softmax_size, conf.top_n)
        self.crit = nn.CrossEntropyLoss()

    def forward(self, entity_repr, target=None):
        pre_pred = self.repr_to_softmax(entity_repr)
        pred = self.proj(pre_pred)
        loss = self.crit(pred, target)
        result = {'loss': loss}
        if not self.training:
            pred_top100 = pred.argsort(dim=1, descending=True)[:, :100]
            result['entity_pred'] = pred_top100
        return result


class EntityHead_continuous(nn.Module):
    def __init__(self, conf, entity_repr_dim):
        super().__init__()
        emb_tensor = torch.load(conf.kb_emb_file)
        self.target_emb = nn.Embedding.from_pretrained(emb_tensor)
        self.emb_dim = self.target_emb.weight.size(1)
        self.reset_emb_idx()
        self.proj_to_emb = nn.Linear(entity_repr_dim, self.emb_dim)
        self.crit = nn.CosineEmbeddingLoss(margin=0.5, reduction='none')

    def forward(self, encoder_repr, target):
        emb_pred = self.encoder_repr_to_emb(encoder_repr)
        emb_target = self.target_emb(target)
        y = torch.ones_like(target).float()
        loss = self.crit(emb_pred, emb_target, y)
        result = {'loss': loss}
        if not self.training:
            D, idxs = self.emb_idx.search(
                emb_pred.detach().cpu().float().numpy(), k=100)
            pred = torch.tensor(idxs).to(target)
            result['entity_pred'] = pred
        return result

    def encoder_repr_to_emb(self, encoder_repr):
        return self.proj_to_emb(encoder_repr)

    def reset_emb_idx(self):
        import faiss
        self.emb_idx = faiss.IndexFlatIP(self.emb_dim)
        self.emb_idx.add(self.target_emb.weight.detach().cpu().numpy())
