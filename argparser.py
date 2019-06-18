import argparse
from pathlib import Path
import json
import os

from dougu import mkdir


def add_generate_graph_args(a):
    a.add_argument(
        '--graph-type', type=str, required=True,
        help='name a networkx graph generator')
    a.add_argument('--n-nodes', type=int, required=True)
    a.add_argument(
        '--graph-generator-args', type=json.loads,
        help='JSON string containing arguments to be passed to the networkx '
        'graph generator', required=True)
    a.add_argument(
        '--n-edge-labels', type=int, required=True,
        help='The number of distinct edge labels')
    a.add_argument(
        '--edge-label-file', type=Path,
        help='File containing one edge label (predicate) per line')
    a.add_argument(
        '--edge-label-distribution', type=str,
        help='The distribution from which edge labels are drawn, '
        'e.g. uniform or power-law')
    a.add_argument(
        '--edge-label-distribution-args', type=json.loads, default='{}',
        help='Arguments to be passed to the random distribution generator')
    a.add_argument('--random-seed', type=int, default=2)
    a.add_argument('--n-paths', type=int, default=100)
    a.add_argument('--n-path-samples', type=int, default=1)
    a.add_argument('--prime-predicates', action='store_true')
    a.add_argument('--unique-targets', action='store_true')


def add_model_args(a):
    a.add_argument('--model', type=str)
    a.add_argument('--model-variant', type=str)
    a.add_argument('--emb-dim', type=int)
    a.add_argument('--n-hidden', type=int, default=1024)
    a.add_argument('--n-layers', type=int, default=2)
    a.add_argument('--dropout', type=float, default=0.0)
    a.add_argument('--n-directions', type=int, choices=[1, 2], default=2)
    a.add_argument('--tie-weights', action='store_true')


def add_training_args(a):
    a.add_argument('--batch-size', type=int, default=128)
    a.add_argument('--optim', type=str, default='adam')
    a.add_argument('--learning-rate', type=float, default=0.001)
    a.add_argument('--momentum', type=float, default=0.0)
    a.add_argument('--weight-decay', type=float, default=0.0)
    a.add_argument('--early-stopping', type=int, default=-1)
    a.add_argument('--early-stopping-burnin', type=int, default=20)
    a.add_argument('--eval-every', type=int, default=10)
    a.add_argument('--first-eval-epoch', type=int, default=10)
    a.add_argument('--max-epochs', type=int, default=100)
    a.add_argument('--max-eval-n-inst', type=int, default=1000)


def add_job_args(a):
    a.add_argument('--configs-per-job', type=int, default=1)
    a.add_argument('--trials-per-config', type=int, default=1)
    a.add_argument("--submit-jobs", action="store_true")
    a.add_argument("--collect-jobs", action="store_true")
    a.add_argument("--print-configs", action="store_true")
    a.add_argument("--job-name", type=str)
    a.add_argument('--jc', type=str, default='gpu-container_g1_dev.default')
    a.add_argument('--ac', type=str, default='d=nvcr-cuda-9.0-cudnn7.2')
    a.add_argument("--results-store", type=str, default="out/results.h5")
    a.add_argument("--inspect-results", action="store_true")
    a.add_argument('--cluster-scheduler', type=str, default='grid_engine')
    a.add_argument('--rundir', type=Path)
    a.add_argument('--runid', type=str)
    a.add_argument('--exp-dir', type=Path)
    a.add_argument('--results-dir', type=Path)
    # a.add_argument("--time", type=str, default="08:00:00")
    # a.add_argument('--group', type=str, default='gcb50180')
    # a.add_argument('--queue', type=str, default='rt_G.small')
    a.add_argument('--cycle-gpus', type=int, default=0)


def add_gpu_args(a):
    a.add_argument('--device', type=str, default='cuda:0')
    a.add_argument('--data-on-cpu', action='store_true')


def add_sweep_args(a):
    a.add_argument('--sweep-n-hidden', type=int, nargs='+')
    a.add_argument('--sweep-n-paths', type=int, nargs='+')
    a.add_argument('--graph-sweep', action='store_true')


def add_plot_args(a):
    a.add_argument('--frame-every-n-epochs', type=int, default=10)
    a.add_argument('--animate', action='store_true')
    a.add_argument('--default-edge-color', type=str, default='grey')
    a.add_argument('--cmap', type=str, default='RdYlBu_r')


def get_argparser():
    desc = 'TODO'
    a = argparse.ArgumentParser(description=desc)
    a.add_argument('command', type=str)
    a.add_argument('--no-commit', action='store_true')
    a.add_argument('--outdir', type=Path, default='out')
    a.add_argument('--exp-name', required=True)
    a.add_argument('--path-sample-id', type=int, default=0)
    a.add_argument('--max-paths', type=int, default=100000)
    a.add_argument('--dataset', type=str)
    add_model_args(a)
    add_generate_graph_args(a)
    add_training_args(a)
    add_gpu_args(a)
    add_job_args(a)
    add_sweep_args(a)
    add_plot_args(a)
    return a


def get_args():
    a = get_argparser()
    args = a.parse_args()
    if 'n' not in args.graph_generator_args:
        args.graph_generator_args['n'] = args.n_nodes
    assert args.graph_generator_args['n'] == args.n_nodes
    is_batchjob = (
        'JOB_SCRIPT' in os.environ and os.environ['JOB_SCRIPT'] != 'QRLOGIN')
    if is_batchjob and 'JOB_ID' in os.environ:
        args.jobid = os.environ['JOB_ID']
    if args.runid is not None:
        args.rundir = mkdir(args.outdir / args.runid)
    else:
        from dougu import next_rundir
        args.rundir = next_rundir()
        args.runid = args.rundir.name
    if args.exp_name:
        args.exp_dir = args.outdir / args.exp_name
        args.results_dir = mkdir(args.exp_dir / 'results.new')
        args.results_store = args.exp_dir / 'results.h5'
    return args
