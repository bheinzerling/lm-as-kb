import os
from itertools import islice
import json

import torch

from ignite.engine import Events

from utils import yesno_mark
from utils.torchutil import count_parameters

from data import Data
from model import KBMemory
from metrics import Average, TopKCategoricalAccuracy

from trainer_base import TrainerBase


class Trainer(TrainerBase):

    def misc_init(self):
        c = self.conf
        is_batchjob = (
            'JOB_SCRIPT' in os.environ and
            os.environ['JOB_SCRIPT'] != 'QRLOGIN')
        if is_batchjob and 'JOB_ID' in os.environ:
            c.jobid = os.environ['JOB_ID']
        c.max_train_inst = c.n_facts
        if c.n_facts / c.batch_size < 50:
            c.batch_size = max(1, int(c.n_facts / 50))
        self.log(f'n_facts: {c.n_facts} | batch size: {c.batch_size}')

    def make_model(self):
        vocab_size = self.data.tokenizer.vocab_size
        padding_idx = self.data.tokenizer.pad_token_id
        if self.conf.entity_repr == 'continuous':
            self.log(self.conf.kb_emb_file)
        return KBMemory.get('kbmemory_' + self.conf.architecture)(
            self.conf, vocab_size, padding_idx)

    @property
    def additional_params_dict(self):
        return {
            'params': [param for param in self.model.entity_head.parameters()],
            'lr': self.conf.predictor_lr}

    def load_data(self):
        self.data = Data.get(self.conf.dataset)(self.conf)
        self.data.log_size()

    def metrics(self, for_eval=False):
        metrics = {'loss': Average(output_transform=lambda r: r['loss'])}

        def output_transform(result):
            return result['entity_pred'], result['entity_target']

        if for_eval:
            for name, k in [('acc', 1), ('hits10', 10), ('hits100', 100)]:
                metrics[name] = TopKCategoricalAccuracy(
                    k=k,
                    output_transform=output_transform,
                    already_sorted=True)
        return metrics

    def make_eval_step(self):
        def eval_step(eval_engine, batch):
            self.model.eval()
            with torch.no_grad():
                if self.is_dist_main():
                    eval_engine.state.batches.append(batch)
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                result = self.model(batch)
                result['loss'] = result['loss'].mean()
                result['entity_target'] = batch['entity_ids']
                if self.is_dist_main():
                    eval_engine.state.outputs.append(result)
                return result
        return eval_step

    @property
    def exp_params(self):
        params = dict(
            entity_repr=self.conf.entity_repr,
            n_hidden=self.conf.n_hidden,
            n_layers=self.conf.n_layers,
            n_params=count_parameters(self.model),
            jobid=getattr(self.conf, 'jobid', -1),
            )
        if self.conf.paraphrase_id is not None:
            params['paraphrase_id'] = self.conf.paraphrase_id
        conf_param_names = [
            'n_facts',
            'top_n',
            'architecture',
            'transformer_model',
            'random_seed',
            'kb_emb_dim',
            'dataset',
            'max_seq_len',
            ]
        for name in conf_param_names:
            params[name] = getattr(self.conf, name)
        return params

    def setup_bookkeeping(self):
        @self.eval_engine.on(Events.STARTED)
        def init_state(engine):
            engine.state.batches = []
            engine.state.outputs = []

        @self.eval_engine.on(Events.COMPLETED)
        def write_predictions(engine):
            if self.conf.dev:
                return
            if self.data.has_train_data():
                epoch = self.train_engine.state.epoch
            else:
                epoch = 0
            fname = f'predictions.e{epoch:04d}.jl'
            outfile = self.conf.rundir / fname
            with outfile.open('w') as out:
                for bo in zip(engine.state.batches, engine.state.outputs):
                    batch, output = bo
                    for c, t, p in self.data.to_context_target_pred(*bo):
                        d = dict(ctx=c, target=t, pred=p, match=int(p == t))
                        line = json.dumps(d)
                        out.write(line + '\n')

        @self.eval_engine.on(Events.COMPLETED)
        def print_examples(engine):
            if self.conf.no_print_examples:
                return
            batch = engine.state.batch
            output = engine.state.output
            ctx_tgt_pred = self.data.to_context_target_pred(batch, output)
            for c, t, p in islice(ctx_tgt_pred, 5):
                mark = yesno_mark(t == p)
                print(f'{c} | {t} | {p} {mark}')

        @self.eval_engine.on(Events.COMPLETED)
        def store_best_eval_score(engine):
            acc = engine.state.metrics['acc']
            if acc > getattr(self.train_engine.state, 'best_acc', 0):
                self.train_engine.state.best_acc = acc

        @self.eval_engine.on(Events.COMPLETED)
        def check_if_memorized(engine):
            acc = engine.state.metrics['acc']
            if acc >= self.conf.memorization_threshold:
                self.log(f'Dataset memorized with acc: {acc}')
                self.train_engine.terminate()
