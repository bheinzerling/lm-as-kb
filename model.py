from torch import nn

from dougu import SubclassRegistry


class PathMemory(nn.Module, SubclassRegistry):
    def __init__(self):
        super().__init__()

    @staticmethod
    def load(conf, data):
        model = PathMemory.get(conf.model)(conf, data)
        if conf.device.startswith('cuda'):
            model.cuda(conf.device)
        # TODO load model checkpoint
        return model


class RnnPathMemory(PathMemory):

    def __init__(self, conf, data):
        super().__init__()
        n_emb = data.tensor.max().item() + 1
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
        self.backward_enc = _make_rnn()
        repr_dim = conf.n_hidden

        self.predictor = nn.Linear(repr_dim, n_emb)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.crit = nn.NLLLoss()

    def forward(self, path, target=None):
        # bw_path, bw_target = batch[:, 1:], batch[:, 0]
        # reverse bw_path
        fw_result = self.predict(self.forward_enc, path, target)
        # bw_pred, bw_loss = self.predict(self.backward_enc, bw_path, bw_target)
        # loss = (fw_loss + bw_loss) / 2
        # return (fw_pred, bw_pred), loss
        return fw_result

    def predict(self, rnn, path, target=None):
        path_emb = self.emb(path)
        rnn_out, rnn_hid = rnn(path_emb)
        path_enc = rnn_out[:, -1]
        return self._predict(path_enc, target)

    def _predict(self, path_enc, target=None):
        logit = self.predictor(path_enc)
        pred = self.log_softmax(logit)
        # result = {'pred': pred}
        if target is not None:
            loss = self.crit(pred, target)
            # result['loss'] = loss
            return pred, loss
        return pred
