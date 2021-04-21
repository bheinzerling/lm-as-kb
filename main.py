import math
import subprocess
import sys

import torch

from utils import args_to_list

from argparser import get_conf
from trainer import Trainer


def get_train_cmd(conf):
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        dist_cmd = [
            'python',
            '-m' 'torch.distributed.launch',
            '--nnodes=1',
            '--node_rank=0',
            f'--nproc_per_node={n_gpus}',
            '--use_env',
            'main.py']
    else:
        dist_cmd = []
    conf.command = 'train'
    cmd = args_to_list(conf, positional_arg='command')
    return dist_cmd + cmd


def find_max_capacity(conf):
    facts_delta = conf.n_facts
    min_facts_delta = conf.n_facts // 10
    last_memorized_n_facts = 0
    max_n_facts = 2 * 10**7
    while conf.n_facts <= max_n_facts:
        torch.cuda.empty_cache()
        conf.runid = conf.rundir = None
        trainer = Trainer(conf)
        if conf.force_new_run:
            runs = None
        else:
            runs = trainer.exp_logger.query_results()

        def get_acc():
            if runs is not None:
                finished_runs = runs[runs.status == 'FINISHED']
                if 'metrics.acc' in runs:
                    return finished_runs['metrics.acc'].max()
            return math.nan
        acc = get_acc()
        if not math.isnan(acc):
            trainer.log(
                f'Found existing run: {conf.n_facts} facts '
                f'with memorization acc {acc}')
        else:
            if runs is not None:
                running_runs = runs[runs.status == 'RUNNING']
                if not running_runs.empty:
                    assert len(running_runs) == 1
                    run_to_resume = running_runs.loc[0]
                    print('resuming run:', run_to_resume)
                    conf.mlflow_runid = run_to_resume.run_id
                    conf.runid = run_to_resume['tags.mlflow.runName']
            cmd = get_train_cmd(conf)
            subprocess.run(cmd, stderr=sys.stderr, stdout=sys.stdout)
            acc = get_acc()
            trainer.log(
                f'New run: {conf.n_facts} facts with memorization acc {acc}')
        if acc >= conf.memorization_threshold:
            last_memorized_n_facts = conf.n_facts
            conf.n_facts += facts_delta
        else:
            trainer.log(f'failed to memorize {conf.n_facts} facts.')
            facts_delta = facts_delta // 2
            if facts_delta < min_facts_delta:
                trainer.log(
                    'facts_delta too small. Last memorized n_facts: '
                    f'{last_memorized_n_facts}')
                break
            conf.n_facts = last_memorized_n_facts + facts_delta
            trainer.log(f'New delta: {facts_delta} | n_facts: {conf.n_facts}')


def paraphrase_train(conf):
    conf.paraphrase_mode = 'train'
    conf.paraphrase_id = 0
    trainer = Trainer(conf)
    result = trainer.train()
    model_file = conf.rundir / result['checkpoint_file']
    print(model_file)


def paraphrase_eval(conf):
    conf.paraphrase_mode = 'finetune'
    if conf.paraphrase_id is not None:
        paraphrase_ids = [conf.paraphrase_id]
    else:
        paraphrase_ids = [0] + conf.paraphrase_ids
    for n_inst in conf.n_finetune_insts:
        conf.n_facts = n_inst
        if n_inst / conf.batch_size < 25:
            conf.batch_size = max(2, n_inst // 25)
        for i in paraphrase_ids:
            conf.paraphrase_id = i
            trainer = Trainer(conf)
            if conf.n_facts > 0:
                trainer.train()
            else:
                trainer.test()
            trainer.exp_logger.log_params({'test_done': 1})


if __name__ == "__main__":
    conf = get_conf()
    if conf.command in globals():
        globals()[conf.command](conf)
    else:
        getattr(Trainer(conf), conf.command)()
