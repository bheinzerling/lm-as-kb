from functools import wraps
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from dougu import SubclassRegistry, flatten


def maybe_cuda(fn):
    @wraps(fn)
    def wrapper(conf, data):
        model = fn(conf, data)
        if conf.device.startswith('cuda'):
            model = model.cuda(conf.device)
        # TODO load model checkpoint
        return model
    return wrapper


class KnowledgeBaseEmbedding(nn.Module, SubclassRegistry):
    @staticmethod
    @maybe_cuda
    def load(conf, data):
        return KnowledgeBaseEmbedding.get(conf.model)(conf, data)


class PathMemory(nn.Module, SubclassRegistry):
    @staticmethod
    @maybe_cuda
    def load(conf, data):
        return PathMemory.get(conf.model)(conf, data)


class LanguageModel(nn.Module, SubclassRegistry):
    @staticmethod
    def load(conf, data):
        model = LanguageModel.get(conf.model)(conf, data)
        if conf.device.startswith('cuda'):
            model = model.cuda(conf.device)
        # TODO load model checkpoint
        return model


class DistMult(KnowledgeBaseEmbedding):

    def __init__(self, conf, data):
        super().__init__()
        n_emb = data.paths.max().item() + 1
        self.n_entities = conf.n_nodes
        self.n_neg_examples = conf.n_neg_examples
        self.n_predicates = conf.n_edge_labels
        assert self.n_entities + self.n_predicates == n_emb
        self.p_emb = nn.Embedding(self.n_predicates, conf.emb_dim)
        self.entity_emb = nn.Embedding(self.n_entities, conf.emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.p_emb.weight)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.ranking_crit = MarginRankingLoss()
        # self.ranking_crit = nn.MarginRankingLoss()
        self.crit = nn.NLLLoss()

    def forward(self, path, target=None):
        s = path[:, 0]
        p = path[:, 1] - self.n_entities
        s_emb = self.entity_emb(s)
        p_emb = self.p_emb(p)
        o_logits = (s_emb * p_emb) @ self.entity_emb.weight.transpose(0, 1)
        pred = self.log_softmax(o_logits)
        if target is None:
            return pred
        o_pos = target
        bs = len(target)
        # create negative examples
        n_neg_idxs = bs * self.n_neg_examples
        neg_idxs = (torch.arange(n_neg_idxs) % bs)[torch.randperm(n_neg_idxs)]
        o_negs = target[neg_idxs].view(bs, self.n_neg_examples)
        o_pos_emb = self.entity_emb(o_pos)
        o_negs_emb = self.entity_emb(o_negs)
        # i = batch dimension index, j = embedding dimension index
        o_pos_scores = torch.einsum('ij,ij,ij->i', s_emb, p_emb, o_pos_emb)
        # k = negative sample index [0..self.n_negative_samples - 1]
        o_negs_scores = torch.einsum('ij,ij,ikj->ik', s_emb, p_emb, o_negs_emb)
        # equivalent to:
        # sp_emb = s_emb * p_emb
        # o_negs_scores = torch.bmm(
        #     sp_emb.unsqueeze(1), o_negs_emb.transpose(1, 2)).squeeze(1)
        loss = self.ranking_crit(o_pos_scores, o_negs_scores)
        # loss = self.ranking_crit(o_pos_scores.unsqueeze(1).expand_as(o_negs_scores), o_negs_scores, torch.ones_like(o_negs_scores))
        return pred, loss


class MarginRankingLoss(nn.Module):
    def forward(self, pos_scores, negs_scores):
        # Eq. 3 in https://arxiv.org/pdf/1412.6575
        margin = negs_scores - pos_scores.unsqueeze(1)
        loss = (margin + 1).clamp_(min=0).mean()
        return loss


class TransE(KnowledgeBaseEmbedding):

    def __init__(self, conf, data):
        super().__init__()
        self.n_entities = conf.n_nodes
        self.n_predicates = conf.n_edge_labels
        self.p_emb = nn.Embedding(self.n_predicates, conf.emb_dim)
        self.entity_emb = nn.Embedding(self.n_entities, conf.emb_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.crit = nn.NLLLoss()

    def forward(self, path, target=None):
        s = path[:, 0]
        p = path[:, 1] - self.n_entities
        s_emb = self.entity_emb(s)
        p_emb = self.p_emb(p)
        o_logits = (s_emb * p_emb) @ self.entity_emb.weight.transpose(0, 1)
        pred = self.log_softmax(o_logits)
        if target is not None:
            loss = self.crit(pred, target)
            return pred, loss
        return pred


class RnnPathMemory(PathMemory):

    def __init__(self, conf, data):
        super().__init__()
        n_emb = data.paths.max().item() + 1
        self.emb = nn.Embedding(n_emb, conf.emb_dim)

        def _make_rnn():
            return getattr(nn, conf.model_variant)(
                input_size=conf.emb_dim,
                hidden_size=conf.n_hidden,
                num_layers=conf.n_layers,
                dropout=conf.dropout if conf.n_layers > 1 else 0.0,
                bidirectional=False,
                batch_first=True)

        self.forward_enc = _make_rnn()
        # self.backward_enc = _make_rnn()
        repr_dim = conf.n_hidden

        self.out = nn.Linear(repr_dim, n_emb)
        if conf.tie_weights:
            self.out.weight = nn.Parameter(self.emb.weight)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.crit = nn.NLLLoss()

    def forward(self, path, target=None):
        fw_result = self.predict(self.forward_enc, path, target)
        return fw_result

    def predict(self, rnn, path, target=None):
        path_emb = self.emb(path)
        rnn_out, rnn_hid = rnn(path_emb)
        path_enc = rnn_out[:, -1]
        return self._predict(path_enc, target)

    def _predict(self, path_enc, target=None):
        logit = self.out(path_enc)
        pred = self.log_softmax(logit)
        if target is not None:
            loss = self.crit(pred, target)
            return pred, loss
        return pred


class TransformerPathMemory(PathMemory):
    #TODO: doesn't learning anything, converges on loss 6.9  acc 0.005
    #TODO: input seq len 2 too short?
    def __init__(self, conf, data):
        super().__init__()
        n_emb = data.paths.max().item() + 1
        self.emb = nn.Embedding(n_emb, conf.n_hidden)
        nn.init.xavier_uniform_(self.emb.weight)
        # path_decoder = TransformerPathMemoryDecoder(conf)
        from transformer import Transformer
        self.transformer = Transformer(
            d_model=conf.n_hidden,
            num_encoder_layers=conf.n_layers,
            num_decoder_layers=1,
            dropout=conf.dropout,
            )

        repr_dim = conf.n_hidden

        self.out = nn.Linear(repr_dim, n_emb)
        if conf.tie_weights:
            self.out.weight = nn.Parameter(self.emb.weight)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.crit = nn.NLLLoss()

    def forward(self, path, target=None):
        # transformer is not batch_first
        path_emb = self.emb(path).transpose(0, 1)
        if target is None:
            target_emb = path_emb
        else:
            assert target.dim() == 1
            target_emb = self.emb(target.unsqueeze(0))
        pred_emb = self.transformer(path_emb, target_emb)[0]
        return self._predict(pred_emb, target)

    def _predict(self, path_enc, target=None):
        logit = self.out(path_enc)
        pred = self.log_softmax(logit)
        if target is not None:
            loss = self.crit(pred, target)
            return pred, loss
        return pred


class RnnLanguageModel(LanguageModel):

    def __init__(self, conf, data):
        super().__init__()
        n_emb = max(path.max().item() for path in data.paths) + 1
        self.emb = nn.Embedding(n_emb, conf.emb_dim)

        def _make_rnn():
            return getattr(nn, conf.model_variant)(
                input_size=conf.emb_dim,
                hidden_size=conf.n_hidden,
                num_layers=conf.n_layers,
                dropout=conf.dropout if conf.n_layers > 1 else 0.0,
                bidirectional=False,
                batch_first=True)

        self.forward_enc = _make_rnn()
        # self.backward_enc = _make_rnn()
        repr_dim = conf.n_hidden

        self.out = nn.Linear(repr_dim, n_emb)
        if conf.tie_weights:
            self.out.weight = nn.Parameter(self.emb.weight)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.crit = nn.NLLLoss()

    def forward(self, path, target=None):
        path_padded = pad_sequence(path, batch_first=True, padding_value=-1)
        breakpoint()
