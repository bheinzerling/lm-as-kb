from argparser import get_args

import torch
from dougu.ignite import Engine, Events
from dougu.ignite.metrics import Loss, Accuracy
from dougu.ignite.handlers import EarlyStopping

from dougu import json_dump, dump_args, autocommit
from dougu.torchutil import get_optim, set_random_seed
from data import KnowledgeGraphPaths
from model import PathMemory


def make_trainer(model, optimizer):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        fw_path, fw_target = batch[:, :-1], batch[:, -1]
        fw_pred, fw_loss = model(fw_path, fw_target)
        fw_loss.backward()
        optimizer.step()
        return fw_loss.item()

    return Engine(_update, name='trainer')


def make_evaluator(model, metrics):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            fw_path, fw_target = batch[:, :-1], batch[:, -1]
            fw_pred = model(fw_path)
            return fw_pred, fw_target

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def train(conf):
    data = KnowledgeGraphPaths.load(conf)
    model = PathMemory.load(conf, data)
    optim = get_optim(conf, model)

    trainer = make_trainer(model, optim)
    trainer.log('rundir ' + str(conf.rundir))
    metrics = {'nll': Loss(model.crit), 'acc': Accuracy()}
    evaluator = make_evaluator(model, metrics)

    def score_function(engine):
        return engine.state.metrics['acc']

    if conf.early_stopping > 0:
        handler = EarlyStopping(
            patience=conf.early_stopping,
            score_function=score_function,
            trainer=trainer,
            burnin=conf.early_stopping_burnin)
        evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(data.loader_trainval)
        metrics = evaluator.state.metrics
        trainer.log("epoch {:04d} {} | loss {:.4f} | acc {:.4f}".format(
            trainer.state.epoch,
            trainer.state.last_epoch_duration,
            metrics['nll'],
            metrics['acc']))
        trainer.state.last_acc = metrics['acc']

    @evaluator.on(Events.COMPLETED)
    def check_if_memorized(evaluator):
        acc = evaluator.state.metrics['acc']
        if acc > 0.9999:
            trainer.log('Dataset memorized.')
            trainer.terminate()

    @trainer.on(Events.COMPLETED)
    def save_results(trainer):
        epoch = trainer.state.epoch
        result_keys = [
            'runid',
            'graph_type',
            'unique_targets',
            'n_nodes',
            'n_edge_labels',
            'n_paths',
            'model_variant',
            'n_hidden',
            'n_layers',
            'emb_dim']
        results = {
            k: conf.__dict__[k]
            for k in result_keys}
        results['epoch'] = epoch
        results['acc'] = trainer.state.last_acc
        results_file = conf.results_dir / 'results.json'
        json_dump(results, results_file)
        trainer.log(results_file)

    trainer.run(data.loader, max_epochs=conf.max_epochs)


if __name__ == '__main__':
    conf = get_args()
    dump_args(conf, conf.rundir / 'conf.json')
    set_random_seed(conf.random_seed)
    if not conf.no_commit:
        conf.commit = autocommit(runid=conf.runid)
    globals()[conf.command](conf)
