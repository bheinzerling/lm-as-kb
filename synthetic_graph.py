import random
import json
from collections import defaultdict

import numpy as np
import networkx as nx
from dougu import mkdir

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
    undir_graph = getattr(nx, conf.graph_type)(**conf.graph_generator_args)
    n_edges = len(undir_graph.edges())

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
        edges = [(nonprimes[v], nonprimes[w]) for v, w in undir_graph.edges()]
        edge_labels = primes.take(edge_label_idxs)
    else:
        edges = list(undir_graph.edges())
        edge_labels = edge_label_idxs + conf.n_nodes

    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    edge_idx = 0
    for node, neighbors in graph.adjacency():
        for neighbor, edict in neighbors.items():
            edict['label'] = edge_labels[edge_idx]
            edge_idx += 1
    assert n_edges == edge_idx
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

    def _sample_paths():
        yielded = 0
        graph_nodes = np.array(list(graph.nodes))
        yielded_paths = defaultdict(set)
        if conf.unique_targets:
            while True:
                start_node_idxs = np.random.randint(
                    0, len(graph.nodes), conf.n_paths - yielded)
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
                if yielded >= conf.n_paths:
                    break
        else:
            while True:
                start_node_idxs = np.random.randint(
                    0, len(graph.nodes), conf.n_paths - yielded)
                start_nodes = graph_nodes[start_node_idxs]
                for start_node in start_nodes:
                    neighbors = list(graph.adj[start_node].items())
                    if not neighbors:
                        continue
                    end_node, edge_dict = random.choice(neighbors)
                    yield list(map(
                        int, (start_node, edge_dict['label'], end_node)))
                    yielded += 1
                if yielded >= conf.n_paths:
                    break

    for sample_id in range(conf.n_path_samples):
        paths = list(_sample_paths())
        outfile = get_path_sample_file(conf, sample_id)
        print(outfile)
        with outfile.open('w') as out:
            for path in paths:
                out.write(json.dumps(path) + '\n')


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


def path_sample_conf_str(conf, sample_id):
    parts = [
        'n_paths' + str(conf.n_paths),
        f'sample_id{sample_id:02d}']
    return ('unique_targets.' if conf.unique_targets else '') + '.'.join(parts)


def get_path_sample_file(conf, sample_id):
    conf_str = path_sample_conf_str(conf, sample_id)
    return get_graphdir(conf) / (conf_str + '.jl')


if __name__ == '__main__':
    conf = get_args()
    globals()[conf.command](conf)
