from collections import defaultdict

import torch

from dougu.ignite import Engine, Events
from dougu.ignite.metrics import Loss, Accuracy
from dougu.ignite.handlers import EarlyStopping, ModelCheckpoint
from dougu import json_dump, dump_args, autocommit
from dougu.torchutil import get_optim, set_random_seed

from data import KnowledgeGraphPaths
from model import PathMemory, LanguageModel
from argparser import get_args


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
        io = engine.state.io
        with torch.no_grad():
            fw_path, fw_target = batch[:, :-1], batch[:, -1]
            fw_pred = model(fw_path)
            fw_pred_prob = torch.exp(fw_pred)
            fw_correct_prob = fw_pred_prob.gather(
                1, fw_target.unsqueeze(1))
            fw_max_prob = fw_pred_prob.max(dim=1)[0]
            fw_prob_ratio = fw_correct_prob.squeeze(1) / fw_max_prob
            fw_rank = (fw_pred_prob >= fw_correct_prob).sum(dim=1)
            io['fw_path'].append(fw_path.cpu())
            io['fw_target'].append(fw_target.cpu())
            io['fw_prob_ratio'].append(fw_prob_ratio.cpu())
            io['fw_rank'].append(fw_rank.cpu())
            return fw_pred, fw_target

    engine = Engine(_inference)

    @engine.on(Events.STARTED)
    def reset_io(engine):
        engine.state.io = defaultdict(list)

    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def memorize_paths(conf):
    data = KnowledgeGraphPaths.load(conf)
    model = PathMemory.load(conf, data)
    optim = get_optim(conf, model)
    if conf.learning_rate_scheduler == "plateau":
        optimum = 'max'
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(
            optim, factor=0.5, patience=10, mode=optimum,
            verbose=True)
    elif conf.learning_rate_scheduler:
        raise ValueError(
            "Unknown lr_scheduler: " + conf.learning_rate_scheduler)
    else:
        lr_scheduler = None

    trainer = make_trainer(model, optim)
    trainer.log('rundir ' + str(conf.rundir))
    metrics = {'nll': Loss(model.crit), 'acc': Accuracy()}
    # TODO: add MRR metric, run on YAGO3_10
    evaluator = make_evaluator(model, metrics)
    checkpointer = ModelCheckpoint(
        conf.rundir, 'final',
        score_name='acc',
        score_function=lambda _: evaluator.state.metrics['acc'],
        n_saved=1)
    trainer.add_event_handler(
        Events.COMPLETED, checkpointer, {'model': model})

    def score_function(engine):
        return -int(engine.state.metrics['nll'] * 1000)

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

    if lr_scheduler is not None:
        @evaluator.on(Events.COMPLETED)
        def scheduler_step(evaluator):
            lr_scheduler.step(evaluator.state.metrics['acc'])

    if conf.write_predictions:
        @evaluator.on(Events.COMPLETED)
        def write_predictions(evaluator):
            io = {k: torch.cat(v) for k, v in evaluator.state.io.items()}
            assert len(set(map(len, io.values()))) == 1
            epoch = trainer.state.epoch
            outfile = conf.rundir / f'predictions.e{epoch:06d}.pt'
            torch.save(io, outfile)

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
            'model',
            'model_variant',
            'n_hidden',
            'n_layers',
            'emb_dim']
        results = {
            k: conf.__dict__[k]
            for k in result_keys}
        results['epoch'] = epoch
        results['acc'] = trainer.state.last_acc
        fname = conf.runid + '.json' if conf.runid else 'results.json'
        results_file = conf.results_dir / fname
        json_dump(results, results_file)
        trainer.log(results_file)

    trainer.run(data.loader, max_epochs=conf.max_epochs)


def make_lm_trainer(model, optimizer):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        model(batch)
        fw_path, fw_target = batch[:, :-1], batch[:, -1]
        fw_pred, fw_loss = model(fw_path, fw_target)
        fw_loss.backward()
        optimizer.step()
        return fw_loss.item()

    return Engine(_update, name='trainer')


def train_lm(conf):
    data = KnowledgeGraphPaths.load(conf)
    model = LanguageModel.load(conf, data)
    optim = get_optim(conf, model)
    trainer = make_lm_trainer(model, optim)

    trainer.run(data.loader, max_epochs=conf.max_epochs)


if __name__ == '__main__':
    conf = get_args()
    dump_args(conf, conf.rundir / 'conf.json')
    set_random_seed(conf.random_seed)
    if not conf.no_commit:
        conf.commit = autocommit(runid=conf.runid)
    globals()[conf.command](conf)
