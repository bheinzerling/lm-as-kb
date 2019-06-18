from pathlib import Path
from copy import deepcopy
import json
import shutil

from dougu import mkdir, lines
from dougu.gpunode import submit_and_collect

from argparser import get_args


def get_task_configs(_conf):
    conf_idx = 0
    for n_paths in _conf.sweep_n_paths:
        for n_hidden in _conf.sweep_n_hidden:
            if _conf.tie_weights:
                _conf.emb_dim = n_hidden
            conf = deepcopy(_conf)
            conf.graph_generator_args = (
                "'" + json.dumps(conf.graph_generator_args) + "'")
            conf.edge_label_distribution_args = (
                "'" + json.dumps(conf.edge_label_distribution_args) + "'")
            conf.n_paths = n_paths
            conf.n_hidden = n_hidden
            __conf = {}
            for k, v in conf.__dict__.items():
                if isinstance(v, Path):
                    v = v.absolute()
                __conf[k] = v
            if conf.cycle_gpus > 0:
                gpu_id = conf_idx % conf.cycle_gpus
                __conf['device'] = f'cuda:{gpu_id}'
            yield __conf
            conf_idx += 1


def append_finished_results(conf, results):
    mkdir(conf.results_dir)
    print('results dir:', conf.results_dir)
    added_dir = mkdir(conf.results_dir.with_suffix(".added"))
    print('added dir:', added_dir)
    for file in conf.results_dir.iterdir():
        res_lines = list(lines(file))
        if len(res_lines) != 1:
            print("Ignoring malformed file:", file)
            continue
        result = json.loads(res_lines[0])
        results.append(result)
        shutil.move(str(file), str(added_dir))
        print("added:", result)


index = [
    'graph_type',
    'unique_targets',
    'n_nodes',
    'n_edge_labels',
    'n_paths',
    'model_variant',
    'n_hidden',
    'n_layers',
    'emb_dim',
    ]
values = [
    'runid',
    'epoch',
    'acc',
    ]
columns = index + values


def main(conf):
    task_configs = list(get_task_configs(conf))
    submit_and_collect(
        conf, task_configs, index, columns, append_finished_results)


if __name__ == '__main__':
    conf = get_args()
    # globals()[conf.command](conf)
    main(conf)
