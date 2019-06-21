import random
import json
from collections import defaultdict
from itertools import islice

import numpy as np
import networkx as nx
from dougu import mkdir, flatten

from argparser import get_args


# https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188  # NOQA
def primesfrom2to(n):
    """Input n>=6, Returns a array of primes, 2 <= p < n"""
    sieve = np.ones(n//3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n**0.5)//3+1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k*k//3::2*k] = False
            sieve[k*(k-2*(i & 1)+4)//3::2*k] = False
    return np.r_[2, 3, ((3*np.nonzero(sieve)[0][1:]+1) | 1)]


def generate_graph(conf):
    _graph = getattr(nx, conf.graph_type)(**conf.graph_generator_args)
    if conf.graph_type == 'scale_free_graph':
        _graph = nx.DiGraph(_graph)
    n_edges = len(_graph.edges())

    edge_label_noise_generator = np.random.power
    edge_label_noise = edge_label_noise_generator(
        **conf.edge_label_distribution_args, size=n_edges)
    edge_label_idxs = (edge_label_noise * conf.n_edge_labels).astype(int)

    if conf.prime_predicates:
        primes = primesfrom2to(10 * conf.n_edge_labels)
        assert len(primes) >= conf.n_edge_labels
        primes_set = set(primes)
        n_primes = int(
            0.1 * conf.n_nodes + conf.n_nodes / np.log(conf.n_nodes))
        nonprimes = [
            i for i in range(conf.n_nodes + n_primes) if i not in primes_set]
        assert len(nonprimes) >= conf.n_nodes
        edges = [(nonprimes[v], nonprimes[w]) for v, w in _graph.edges()]
        edge_labels = primes.take(edge_label_idxs)
    else:
        edges = list(_graph.edges())
        edge_labels = edge_label_idxs + conf.n_nodes

    if conf.graph_type == 'scale_free_graph':
        graph = _graph
    else:
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
    edge_idx = 0
    for node, neighbors in graph.adjacency():
        for neighbor, edict in neighbors.items():
            edict['label'] = edge_labels[edge_idx]
            edge_idx += 1
    assert n_edges == edge_idx, breakpoint()
    return graph


def sample_paths(conf, graph=None):
    np.random.seed(conf.random_seed)
    random.seed(conf.random_seed)
    if graph is None:
        graph = generate_graph(conf)
    outdir = get_graphdir(conf)
    graphfile = get_graphfile(outdir)
    nx.write_edgelist(graph, graphfile)
    print(graphfile)

    if conf.max_path_len == conf.min_path_len == 3:
        sample_fn = _sample_triples
    else:
        sample_fn = _sample_paths

    for sample_id in range(conf.n_path_samples):
        paths = islice(sample_fn(conf, graph), conf.max_paths)
        outfile = get_path_sample_file(conf, sample_id)
        print(outfile)
        with outfile.open('w') as out:
            for path in paths:
                out.write(json.dumps(path) + '\n')


def _sample_paths(conf, graph):
    from collections import Counter
    yielded = 0
    graph_nodes = np.array(list(graph.nodes))
    yielded_paths = set()

    path_lengths = np.random.poisson(
        lam=conf.path_len_poisson_lambda,
        size=conf.max_paths) * 2 + conf.min_path_len
    path_lengths = path_lengths.clip(0, conf.max_path_len)
    print(sorted(Counter(path_lengths).most_common()))
    while True:
        failures = 0
        start_node_idxs = np.random.randint(
            0, len(graph.nodes), conf.max_paths - yielded)
        start_nodes = graph_nodes[start_node_idxs]
        for start_node, path_length in zip(start_nodes, path_lengths):
            path = sample_path(start_node, graph, path_length)
            if path and path not in yielded_paths:
                yield path
                yielded_paths.add(path)
                failures = 0
            else:
                failures += 1
        if failures > 100000:
            raise ValueError(
                f'Cannot sample {conf.max_paths} paths. '
                f'Yielded {yielded} unique paths.')


def sample_path(start_node, graph, path_length, allow_cycles=False):
    path = [(None, int(start_node))]
    if allow_cycles:
        raise NotImplementedError
    visited_nodes = set()
    failures = 0
    while len(tuple(flatten(path))[1:]) < path_length:
        if not path or failures > 10 * path_length:
            return None
        edge_label, node = path[-1]
        visited_nodes.add(int(node))
        adj_entries = graph.adj.get(node, {})
        neighbors = list(adj_entries.items())
        unvisited_neighbors = set(adj_entries.keys()) - visited_nodes
        if not neighbors or not unvisited_neighbors:
            del path[-1]
            failures += 1
            continue
        next_node, next_edge_dict = random.choice(neighbors)
        if next_node in visited_nodes:
            failures += 1
            continue
        path.append((int(next_edge_dict['label']), int(next_node)))
    path = tuple(flatten(path))[1:]
    assert path_length <= len(path) <= path_length + 1, breakpoint()
    return path


def _sample_triples(conf, graph):
    yielded = 0
    graph_nodes = np.array(list(graph.nodes))
    yielded_paths = defaultdict(set)
    if conf.unique_targets:
        failures = 0
        while True:
            start_node_idxs = np.random.randint(
                0, len(graph.nodes), conf.max_paths - yielded)
            start_nodes = graph_nodes[start_node_idxs]
            for start_node in start_nodes:
                neighbors = list(graph.adj[start_node].items())
                if not neighbors:
                    continue
                end_node, edge_dict = random.choice(neighbors)
                label = edge_dict['label']
                if label not in yielded_paths[start_node]:
                    yield list(map(
                        int, (start_node, label, end_node)))
                    yielded_paths[start_node].add(label)
                    yielded += 1
                    failures = 0
                else:
                    failures += 1
                    if failures > 100000:
                        raise ValueError(
                            f'Cannot sample {conf.max_paths} paths. '
                            f'Yielded {yielded} unique paths.')
            if yielded >= conf.max_paths:
                break
    else:
        while True:
            start_node_idxs = np.random.randint(
                0, len(graph.nodes), conf.max_paths - yielded)
            start_nodes = graph_nodes[start_node_idxs]
            for start_node in start_nodes:
                neighbors = list(graph.adj[start_node].items())
                if not neighbors:
                    continue
                end_node, edge_dict = random.choice(neighbors)
                yield list(map(
                    int, (start_node, edge_dict['label'], end_node)))
                yielded += 1
            if yielded >= conf.max_paths:
                break


def graph_conf_str(conf):

    def dict2str(d):
        return '_'.join(f'{k}{v}' for k, v in d.items())

    parts = [
        conf.graph_type,
        dict2str(conf.graph_generator_args),
        'n_nodes' + str(conf.n_nodes),
        'n_edge_labels' + str(conf.n_edge_labels),
        'edge_label_distribution_args' + dict2str(
            conf.edge_label_distribution_args),
        'random_seed' + str(conf.random_seed)]
    return '.'.join(parts) + (
        '.prime_predicates' if conf.prime_predicates else '')


def get_graphdir(conf):
    conf_str = graph_conf_str(conf)
    return mkdir(conf.outdir / conf_str)


def get_graphfile(outdir):
    graphfile = outdir / 'graph.edgelist'
    return graphfile


def _graphfile(conf):
    outdir = get_graphdir(conf)
    return get_graphfile(outdir)


def path_sample_conf_str(conf, sample_id):
    parts = [
        'max_paths' + str(conf.max_paths),
        f'sample_id{sample_id:02d}']
    return ('unique_targets.' if conf.unique_targets else '') + '.'.join(parts)


def get_path_sample_file(conf, sample_id):
    conf_str = path_sample_conf_str(conf, sample_id)
    return get_graphdir(conf) / (conf_str + '.jl')


if __name__ == '__main__':
    conf = get_args()
    globals()[conf.command](conf)
