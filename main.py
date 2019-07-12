from collections import defaultdict

import torch

from dougu.ignite import Engine, Events
from dougu.ignite.metrics import (
    Loss, Accuracy, MeanReciprocalRank, TopKCategoricalAccuracy)
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
        # if hasattr(model, 'to_entity_emb_ids'):
        #     fw_target = model.to_entity_emb_ids(fw_target).view(-1)
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
            # if hasattr(model, 'to_entity_emb_ids'):
            #     fw_target = model.to_entity_emb_ids(fw_target).view(-1)
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
    if conf.inspect:
        breakpoint()
        return

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
    metrics = {
        'nll': Loss(model.crit),
        'acc': Accuracy(),
        'p@10': TopKCategoricalAccuracy(k=10),
        'mrr': MeanReciprocalRank()}
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
        metrics_str = ' | '.join(
            f'{metric} {val:.4f}' for metric, val in metrics.items())
        trainer.log("epoch {:04d} {} | {}".format(
                trainer.state.epoch,
                trainer.state.last_epoch_duration,
                metrics_str))
        trainer.state.last_acc = metrics['acc']
        trainer.state.last_p10 = metrics['p@10']
        trainer.state.last_mrr = metrics['mrr']

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
        results['p@10'] = trainer.state.last_p10
        results['mrr'] = trainer.state.last_mrr
        fname = conf.runid + '.json' if conf.runid else 'results.json'
        results_file = conf.results_dir / fname
        json_dump(results, results_file)
        trainer.log(results_file)

    trainer.run(data.loader, max_epochs=conf.max_epochs)


def plot_emb(conf):
    from dougu.plot import plot_embeddings_bokeh
    assert conf.model_file
    model_dict = torch.load(conf.model_file)
    if 'entity_emb.weight' in model_dict:
        e_emb = model_dict['entity_emb.weight']
        p_emb = model_dict['p_emb.weight']
        emb = torch.cat([e_emb, p_emb], dim=0)
    else:
        emb = model_dict['emb.weight']
    emb = emb.cpu().detach().numpy()
    labels = list(map(str, range(emb.shape[0])))
    ids = list(range(emb.shape[0]))
    classes = ['s'] * conf.n_nodes + ['p'] * conf.n_edge_labels
    outfile = conf.rundir / 'emb.bokeh.html'
    plot_embeddings_bokeh(
        emb, labels=labels, color=ids, outfile=outfile)
    print(outfile)


def plot_acc_correlations(conf):
    from synthetic_graph import get_graph
    from dougu.plot import Figure, plt
    import seaborn as sns
    import numpy as np
    data = KnowledgeGraphPaths.load(conf)
    model = PathMemory.load(conf, data)
    metrics = {
        'nll': Loss(model.crit),
        'acc': Accuracy(),
        'p@10': TopKCategoricalAccuracy(k=10),
        'mrr': MeanReciprocalRank()}
    evaluator = make_evaluator(model, metrics)
    evaluator.run(data.loader_trainval)
    io = {k: torch.cat(v) for k, v in evaluator.state.io.items()}
    graph = get_graph(conf)

    pr = graph.pagerank()
    dg = graph.degree()
    # reciprocal_rank = 1 / io['fw_rank'].float().cpu().numpy()
    rank = io['fw_rank'].float().cpu().numpy()

    for nodes_name, nodes in zip(
            ['source', 'target'], [io['fw_path'][:, 0], io['fw_target']]):
        pagerank_scores = [pr[node] for node in nodes]
        degree = [dg[node] for node in io['fw_target']]
        pagerank = np.argsort(pagerank_scores)
        assert len(pagerank) == len(rank)
        with Figure(
                f'rank_vs_{nodes_name}_pagerank',
                xlabel=f'{nodes_name} pagerank', ylabel='prediction rank'):
            sns.scatterplot(x=pagerank, y=rank)
        with Figure(
                f'rank_vs_{nodes_name}_degree',
                xlabel=f'{nodes_name} pagerank', ylabel='prediction rank'):
            sns.scatterplot(x=degree, y=rank)
    breakpoint()


def fuzz_preact(conf):
    from synthetic_graph import get_graph
    import pandas as pd
    from dougu.plot import Figure, colors, plt
    import seaborn as sns
    import random
    from matplotlib.colors import LogNorm

    random.seed(conf.random_seed)

    data = KnowledgeGraphPaths.load(conf)
    model = PathMemory.load(conf, data)

    def sample_path(graph, target_node):
        incoming = graph.incident(target_node, mode='in')
        if not incoming:
            return sample_path(graph, target_node + 1)
        edge = graph.es[random.choice(incoming)]
        s = edge.source
        p = int(edge['label'])
        return torch.tensor([s, p]).to(device=conf.device), target_node

    # path_idx = 0
    with torch.no_grad():
        graph = get_graph(conf)
        nodes = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 20, 30, 50,
            100, 300, 500, 1000, 3000, 5000, 9000)
        nodes = list(sorted(
            set(nodes) |
            set(random.sample(list(range(conf.n_nodes)), 50))))
        # nodes = (0, 5, 50, 500, 5000)
        stds = (
            0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
            0.25, 0.3, 0.4, 0.5, 1.0, 1.5)
        # pageranks = graph.pagerank(nodes)
        pageranks_stds_same_ratios = []
        for node in nodes:

            path, target_node = sample_path(graph, target_node=node)
            node = target_node
            pagerank = graph.pagerank()[node]
            print(node, pagerank)
            path_emb = model.emb.weight[path].unsqueeze(0)
            rnn_out, rnn_hid = model.forward_enc(path_emb)
            path_enc = rnn_out[:, -1]
            logit = model.out(path_enc)
            pred = model.log_softmax(logit)
            pred_idx = pred.max(dim=1)[1].item()
            # std = path_enc.std()

            size = torch.Size((conf.samples_per_fuzz, path_enc.size(1)))
            mean = torch.zeros(size).to(path_enc)
            # fuzzed_path_enc_std_same_pred_ratio = []
            for _std in stds:
                std = torch.zeros(size).to(path_enc) + _std
                noise = torch.normal(mean, std).to(path_enc)
                fuzzed_path_enc = path_enc + noise
                fuzzed_logit = model.out(fuzzed_path_enc)
                fuzzed_pred = model.log_softmax(fuzzed_logit)
                fuzzed_pred_idx = fuzzed_pred.max(dim=1)[1]
                same_pred = fuzzed_pred_idx == pred_idx
                same_ratio = same_pred.sum().float().item() / len(fuzzed_pred_idx)
                # std2same_ratio[_std] = same_ratio
                pageranks_stds_same_ratios.append({
                    'node': node,
                    'pagerank': pagerank,
                    'indegree': len(graph.incident(node, mode='in')),
                    'std': _std,
                    'same_ratio': same_ratio})
                # same_ratio = torch.zeros(size[0]).to(path_enc) + same_ratio
                # fuzzed_path_enc_std_same_pred_ratio.append((
                #     fuzzed_path_enc, std[:, 0], same_pred, same_ratio))
            # fuzzed_path_enc, std, same_pred, same_ratio = list(map(
                # torch.cat, zip(*fuzzed_path_enc_std_same_pred_ratio)))
        # from dougu.plot import plot_embeddings_bokeh
        df = pd.DataFrame(
            pageranks_stds_same_ratios,
            columns=['node', 'pagerank', 'indegree', 'std', 'same_ratio'])
        # norm = LogNorm(vmax=df.pagerank.min(), vmin=df.pagerank.max())
        with Figure(
                'fuzz_preact_pagerank_std_same_ratio',
                figwidth=12):
            # for node, color in zip(sorted(df.node.unique()), colors):
            #     _df = df[df.node == node]
            #     pagerank = _df.pagerank.tolist()[0]
            #     sns.lineplot(
            #         x=_df['std'],
            #         y=_df.same_ratio,
            #         color=_df.pagerank,
            #         norm=norm,
            #         label=f'Node{node} {pagerank:.6f}')
            # plt.legend(loc='upper right')
            sns.lineplot(data=df, x='std', y='same_ratio', hue='pagerank')

        with Figure(
                'fuzz_preact_indegree_std_same_ratio',
                figwidth=12):
            # for node, color in zip(sorted(df.node.unique()), colors):
            #     _df = df[df.node == node]
            #     pagerank = _df.pagerank.tolist()[0]
            #     sns.lineplot(
            #         x=_df['std'],
            #         y=_df.same_ratio,
            #         color=_df.pagerank,
            #         norm=norm,
            #         label=f'Node{node} {pagerank:.6f}')
            # plt.legend(loc='upper right')
            sns.lineplot(data=df, x='std', y='same_ratio', hue='indegree')


def fuzz_subj_emb(conf):
    from synthetic_graph import get_graph
    import pandas as pd
    from dougu.plot import Figure, colors, plt
    import seaborn as sns
    import random

    random.seed(conf.random_seed)

    def sample_path(graph, target_node):
        incoming = graph.incident(target_node, mode='in')
        if not incoming:
            return sample_path(graph, target_node + 1)
        edge = graph.es[random.choice(incoming)]
        s = edge.source
        p = int(edge['label'])
        return torch.tensor([s, p]).to(device=conf.device), target_node

    data = KnowledgeGraphPaths.load(conf)
    model = PathMemory.load(conf, data)
    graph = get_graph(conf)

    # path_idx = 0
    with torch.no_grad():
        nodes = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 20, 30, 50,
            100, 300, 500, 1000, 3000, 5000, 9000)
        nodes = list(sorted(
            set(nodes) |
            set(random.sample(list(range(conf.n_nodes)), 50))))
        # nodes = (0, 5, 50, 500, 5000)
        stds = (
            0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
            0.25, 0.3, 0.4, 0.5, 1.0, 1.5)
        # pageranks = graph.pagerank(nodes)
        pageranks_stds_same_ratios = []
        for node in nodes:
            path, target_node = sample_path(graph, target_node=node)
            node = target_node
            pagerank = graph.pagerank()[node]
            print(node, pagerank)
            path_emb = model.emb.weight[path].unsqueeze(0)
            rnn_out, rnn_hid = model.forward_enc(path_emb)
            path_enc = rnn_out[:, -1]
            logit = model.out(path_enc)
            pred = model.log_softmax(logit)
            pred_idx = pred.max(dim=1)[1].item()
            # std = path_enc.std()

            size = torch.Size((conf.samples_per_fuzz, 1, path_emb.size(2)))
            mean = torch.zeros(size).to(path_enc)
            # fuzzed_path_enc_std_same_pred_ratio = []
            for _std in stds:
                std = torch.zeros(size).to(path_enc) + _std
                noise = torch.normal(mean, std).to(path_enc)
                breakpoint()
                # TODO fuzz emb
                fuzzed_path_enc = path_enc + noise
                fuzzed_logit = model.out(fuzzed_path_enc)
                fuzzed_pred = model.log_softmax(fuzzed_logit)
                fuzzed_pred_idx = fuzzed_pred.max(dim=1)[1]
                same_pred = fuzzed_pred_idx == pred_idx
                same_ratio = same_pred.sum().float().item() / len(fuzzed_pred_idx)
                pageranks_stds_same_ratios.append({
                    'node': node,
                    'pagerank': pagerank,
                    'indegree': len(graph.incident(node, mode='in')),
                    'std': _std,
                    'same_ratio': same_ratio})
        df = pd.DataFrame(
            pageranks_stds_same_ratios,
            columns=['node', 'pagerank', 'indegree', 'std', 'same_ratio'])
        with Figure(
                'fuzz_subj_emb_pagerank_std_same_ratio',
                figwidth=12):
            sns.lineplot(data=df, x='std', y='same_ratio', hue='pagerank')

        with Figure(
                'fuzz_subj_emb_indegree_std_same_ratio',
                figwidth=12):
            sns.lineplot(data=df, x='std', y='same_ratio', hue='indegree')


def count_neighbors(conf):
    from synthetic_graph import get_graph
    import numpy as np
    import pandas as pd
    from dougu.plot import Figure, colors, plt
    import seaborn as sns
    import random
    data = KnowledgeGraphPaths.load(conf)
    model = PathMemory.load(conf, data)
    graph = get_graph(conf)
    emb = model.emb.weight.cpu().detach().numpy()
    dist_levels = np.array([
        6, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 15, 20])
    pageranks = graph.pagerank()
    nodes = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 20, 30, 50,
        100, 300, 500, 1000, 3000, 5000, 9000]
    pageranks = [graph.pagerank()[node] for node in nodes]

    if len(nodes) > len(emb) / 4:
        dist = np.sqrt(((emb - emb[:, np.newaxis, :])**2).sum(axis=2))
        dist = dist[nodes]
    else:
        dist = np.stack([
            np.sqrt(((emb - emb[node])**2).sum(axis=1))
            for node in nodes], axis=0)

    n_neighbors = (dist[:, None] < dist_levels[None, :, None]).sum(2)
    df = pd.DataFrame([
        {'pagerank': pr, 'dist_level': d, 'n_neighbors': n}
        for pr, n_neighbor in zip(pageranks, n_neighbors)
        for d, n in zip(dist_levels, n_neighbor)])
    with Figure('count_neighbors', figwidth=12):
        sns.lineplot(data=df, x='dist_level', y='n_neighbors', hue='pagerank')
        plt.legend()


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


def plot_degree_distributions(conf):
    from synthetic_graph import get_graph
    from collections import Counter
    from dougu.plot import Figure, colors, plt
    import seaborn as sns
    import numpy as np

    conf.graph_type = 'barabasi'
    conf.graph_generator_args = {'power': 1.1, 'n': 250000}
    barabasi_graph = get_graph(conf)

    conf.graph_type = 'erdos_renyi'
    conf.graph_generator_args = {'m': 2000001, 'n': 250000}
    erdos_renyi_graph = get_graph(conf)

    conf.dataset = conf.graph_type = 'yago3_10'
    yago_data = KnowledgeGraphPaths.load(conf)

    degrees = {
        'in': {
            'barabasi': barabasi_graph.degree(mode='in'),
            'erdos_renyi': erdos_renyi_graph.degree(mode='in'),
            'yago3_10': list(Counter(
                yago_data.paths[:, 2].tolist()).values())},
        'out': {
            'barabasi': barabasi_graph.degree(mode='out'),
            'erdos_renyi': erdos_renyi_graph.degree(mode='out'),
            'yago3_10': list(Counter(
                yago_data.paths[:, 0].tolist()).values())}}
    for direction in 'in', 'out':
        xmax = 300 if direction == 'out' else 60000
        bins = np.logspace(0, np.log(xmax))
        with Figure(f'degree_distribution_{direction}', figwidth=12):
            for graph_type, ds in degrees[direction].items():
                sns.distplot(
                    ds,
                    label=graph_type,
                    bins=bins,
                    hist_kws=dict(log=True, histtype='step'),
                    kde=False)
            plt.xlim(1, xmax)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(f'{direction}-degree')
            plt.ylabel('frequency')
            plt.legend()
    breakpoint()


if __name__ == '__main__':
    conf = get_args()
    dump_args(conf, conf.rundir / 'conf.json')
    set_random_seed(conf.random_seed)
    if not conf.no_commit:
        conf.commit = autocommit(runid=conf.runid)
    globals()[conf.command](conf)
