import random
import json
from collections import defaultdict
from itertools import islice

import numpy as np
import networkx as nx
import igraph
from tqdm import tqdm

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


def get_degree_seq(conf):
    max_degree = conf.max_degree if conf.max_degree else np.sqrt(conf.n_nodes)
    degree_seq_noise_generator = np.random.power
    degree_seq_noise = degree_seq_noise_generator(
        **conf.degree_seq_distribution_args, size=conf.n_nodes) ** 3
    degree_seq = (degree_seq_noise * (max_degree - 1)).astype(int) + 1
    if degree_seq.sum() % 2:
        degree_seq[np.argmax(degree_seq)] += 1
    return degree_seq.tolist()


def generate_graph(conf):
    # print('generating graph')
    random.seed(conf.random_seed)
    degree_seq = get_degree_seq(conf)
    if conf.graph_type == 'barabasi':
        graph = igraph.Graph.Barabasi(
            directed=True, m=degree_seq, **conf.graph_generator_args)
        # found out that networkx is too slow for generating
        # large (100k+ nodes) scale free graphs
        # for now, generate those with much faster igraph library
        # and then load the generated graph with networkx
        # TODO: check if bokeh can import igraph graphs, consider switching
        # edges = ((e.source, e.target) for e in igraph_graph.es)
        # _graph = nx.DiGraph()
        # _graph.add_edges_from(edges)
    elif conf.graph_type == 'degree_sequence':
        try:
            graph = igraph.Graph.Degree_Sequence(degree_seq, method='vl')
        except igraph._igraph.InternalError as e:
            print(e)
            graph = igraph.Graph.Degree_Sequence(
                degree_seq, degree_seq, method='no_multiple')
        graph.to_directed()
    elif conf.graph_type == 'watts_strogatz':
        graph = igraph.Graph.Watts_Strogatz
        breakpoint()
    elif conf.graph_type == 'erdos_renyi':
        graph = igraph.Graph.Erdos_Renyi(
            directed=True, **conf.graph_generator_args)
    else:
        raise NotImplementedError
    #     _graph = getattr(nx, conf.graph_type)(**conf.graph_generator_args)
    #     if conf.graph_type == 'scale_free_graph':
    #         _graph = nx.DiGraph(_graph)
    n_edges = len(graph.es)

    edge_label_noise_generator = np.random.power
    edge_label_noise = edge_label_noise_generator(
        **conf.edge_label_distribution_args, size=n_edges)
    edge_label_idxs = (edge_label_noise * conf.n_edge_labels).astype(int)

    # if conf.prime_predicates:
    #     primes = primesfrom2to(10 * conf.n_edge_labels)
    #     assert len(primes) >= conf.n_edge_labels
    #     primes_set = set(primes)
    #     n_primes = int(
    #         0.1 * conf.n_nodes + conf.n_nodes / np.log(conf.n_nodes))
    #     nonprimes = [
    #         i for i in range(conf.n_nodes + n_primes) if i not in primes_set]
    #     assert len(nonprimes) >= conf.n_nodes
    #     edges = [(nonprimes[v], nonprimes[w]) for v, w in _graph.edges()]
    #     edge_labels = primes.take(edge_label_idxs)
    # else:
    #     # edges = list(_graph.edges())
    edge_labels = edge_label_idxs + conf.n_nodes

    # if conf.graph_type in {'scale_free_graph', 'barabasi'}:
    #     graph = _graph
    # else:
    #     graph = nx.DiGraph()
    #     graph.add_edges_from(edges)
    # edge_idx = 0
    # for node, neighbors in graph.adjacency():
    #     for neighbor, edict in neighbors.items():
    #         edict['label'] = edge_labels[edge_idx]
    #         edge_idx += 1
    for edge_idx, edge in enumerate(graph.es):
        edge.update_attributes(label=edge_labels[edge_idx])
    # assert n_edges == edge_idx, breakpoint()
    return graph


def sample_paths(conf, graph=None):
    # print(f'sampling {conf.max_paths} paths')
    np.random.seed(conf.random_seed)
    random.seed(conf.random_seed)
    if graph is None:
        outdir = get_graphdir(conf)
        graphfile = get_graphfile(outdir)
        if not graphfile.exists():
            graph = generate_graph(conf)
            graph.write_graphml(str(graphfile))
        print('loading graph from', graphfile)
        graph = nx.read_graphml(
            graphfile,
            node_type=lambda n: int(n[1:]))
        for _, _, edge_dict in graph.edges(data=True):
            edge_dict['label'] = int(edge_dict['label'])
        print(graphfile)

    if conf.max_path_len == conf.min_path_len == 3:
        sample_fn = _sample_triples
    else:
        sample_fn = _sample_paths

    for sample_id in range(conf.n_path_samples):
        paths = tqdm(
            islice(sample_fn(conf, graph), conf.max_paths),
            total=conf.max_paths)
        outfile = get_path_sample_file(conf, sample_id)
        print(outfile)
        with outfile.open('w') as out:
            for path in paths:
                out.write(json.dumps(path) + '\n')


def _sample_paths(conf, graph):
    from collections import Counter
    yielded = 0
    graph_nodes = np.array(list(graph.nodes))
    pagerank = nx.pagerank(graph)
    node_idxs = np.arange(len(graph.nodes))
    pageranks = np.array([pagerank[node_idx] for node_idx in node_idxs])
    yielded_paths = set()
    if conf.add_path_end_marker:
        path_end_marker = conf.n_nodes + conf.n_edge_labels
    else:
        path_end_marker = None

    # started_nodes = set()
    # all_nodes = set(node_idxs.tolist())

    def start_node_and_path_lengths_iter():
        p = pageranks
        from scipy.special import softmax
        while True:
            start_node_idxs = np.random.choice(
                node_idxs, size=conf.max_paths, p=p)
            # not_started_nodes = np.array(list(all_nodes - started_nodes))
            # start_node_idxs = np.concatenate([
            #     start_node_idxs, not_started_nodes])
            path_lengths = np.random.poisson(
                lam=conf.path_len_poisson_lambda,
                size=len(start_node_idxs)) * 2 + conf.min_path_len
            path_lengths = path_lengths.clip(0, conf.max_path_len)
            print(sorted(Counter(path_lengths).most_common()))
            p = softmax(p ** 2)
            start_nodes = graph_nodes[start_node_idxs]
            yield from zip(start_nodes, path_lengths)

    start_node_and_path_lengths = start_node_and_path_lengths_iter()
    failures = 0
    print('graph has', len(graph.edges), 'edges')
    while True:
        start_node, path_length = next(start_node_and_path_lengths)
        path = sample_path(
            start_node, graph, path_length,
            path_end_marker=path_end_marker)
        if path and path not in yielded_paths:
            yield path
            # started_nodes.add(path[0])
            yielded_paths.add(path)
            failures = 0
        else:
            failures += 1
        if failures > 1000:
            raise ValueError(
                f'Cannot sample {conf.max_paths} paths. '
                f'Yielded {yielded} unique paths.')


def sample_path(
        start_node, graph, path_length,
        allow_cycles=False,
        path_end_marker=None):
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
    path = list(flatten(path))[1:]
    assert path_length <= len(path) <= path_length + 1, breakpoint()
    if path_end_marker is not None:
        path.append(path_end_marker)
    return tuple(path)


def _sample_triples(conf, graph):
    edges = list(graph.edges(data=True))
    print(len(edges))
    assert conf.max_paths <= len(edges)
    random.shuffle(edges)
    all_triples = map(lambda edge: (edge[0], edge[2]['label'], edge[1]), edges)
    if conf.unique_targets:
        seen_source_preds = defaultdict(set)

        def _filter(triples):
            for triple in triples:
                source, pred, target = triple
                if pred in seen_source_preds[source]:
                    # print('not unique', source, pred)
                    continue
                yield triple
                seen_source_preds[source].add(pred)
        triples = _filter(all_triples)
    else:
        triples = all_triples
    yield from islice(triples, conf.max_paths)


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
        'degree_seq_distribution_args' + dict2str(
            conf.degree_seq_distribution_args),
        'random_seed' + str(conf.random_seed)]
    return '.'.join(parts) + (
        '.prime_predicates' if conf.prime_predicates else '')


def get_graphdir(conf):
    conf_str = graph_conf_str(conf)
    return mkdir(conf.outdir / conf_str)


def get_graphfile(outdir):
    graphfile = outdir / 'graph.graphml'
    return graphfile


def _graphfile(conf):
    outdir = get_graphdir(conf)
    return get_graphfile(outdir)


def path_sample_conf_str(conf, sample_id):
    return _path_sample_conf_str(
        conf, sample_id, conf.min_path_len, conf.max_path_len)


def _path_sample_conf_str(conf, sample_id, min_path_len, max_path_len):
    parts = [
        'max_paths' + str(conf.max_paths),
        'min_path_len' + str(min_path_len),
        'max_path_len' + str(max_path_len),
        f'sample_id{sample_id:02d}']
    return (
        ('unique_targets.' if conf.unique_targets else '') +
        ('path_end_marker.' if conf.add_path_end_marker else '') +
        '.'.join(parts))


def get_path_sample_file(conf, sample_id):
    conf_str = path_sample_conf_str(conf, sample_id)
    return get_graphdir(conf) / (conf_str + '.jl')


def get_triple_sample_file(conf, sample_id):
    conf_str = _path_sample_conf_str(
        conf, sample_id, min_path_len=3, max_path_len=3)
    return get_graphdir(conf) / (conf_str + '.jl')


if __name__ == '__main__':
    conf = get_args()
    globals()[conf.command](conf)
