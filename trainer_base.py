import sys
import os
import subprocess

from torch.nn.utils import clip_grad_norm_
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver


from utils import (
    get_logger,
    mkdir,
    next_rundir,
    dump_args,
    json_load,
    json_dump,
    )
from utils.decorators import cached_property
from utils.torchutil import (
    set_random_seed,
    get_optim,
    get_lr_scheduler,
    log_param_counts,
    fix_dataparallel_statedict,
    )

from experiment_logger import ExperimentLogger


class TrainerBase():
    """Provides basic functionality for training a model:
        - model setup
        - data loading
        - training loop
        - metrics and logging
        - distributed training
        - checkpointing
    """
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        set_random_seed(conf.random_seed)
        self.setup()

    def setup(self):
        self.logger = get_logger()
        self.log = self.logger.info
        self.log(" ".join(sys.argv))
        self.misc_init()
        self.setup_distributed()
        self.setup_rundir()
        self.log('loading data')
        if not self.is_dist_main():
            dist.barrier()
        else:
            self.load_data()
            if self.conf.distributed:
                dist.barrier()
        if not self.is_dist_main():
            self.load_data()
        self.setup_model()
        self.setup_optim()
        if self.data.has_train_data():
            self.setup_lr_scheduler()
        self.distribute_model()
        if self.is_dist_main():
            self.log_jobid()
            if self.requires_exp_log:
                self.exp_logger = ExperimentLogger(self.conf, self.exp_params)
            self.setup_bookkeeping()
            if self.data.has_train_data():
                self.setup_early_stopping()
                for event, handler in self.event_handlers_train:
                    self.train_engine.add_event_handler(event, handler)

    @property
    def requires_exp_log(self):
        return self.conf.command != 'cache_dataset'

    def misc_init(self):
        pass

    def log_jobid(self):
        """Log the job ID when running on a cluster."""
        jobid = getattr(self.conf, 'jobid', 0)
        if jobid and self.conf.local_rank == 0:
            jobid_file = self.conf.rundir / 'last_jobid'
            with jobid_file.open('w') as out:
                out.write(str(jobid))

    def setup_rundir(self):
        """All data produced during a training run will be written to
        directory.
        """
        c = self.conf
        if self.is_dist_main():
            if c.runid is not None:
                c.rundir = mkdir(c.outdir / c.runid)
            else:
                c.rundir = next_rundir()
                c.runid = c.rundir.name
            c.trainer_state_file = c.rundir / 'trainer_state.json'
            c.checkpointer_state_file = c.rundir / 'checkpointer_state.pt'
            dump_args(c, c.rundir / 'conf.json')
            self.log(f'rundir: {c.rundir}')

    def setup_distributed(self):
        """Setup distributed training.
        """
        self.conf.distributed = 'LOCAL_RANK' in os.environ
        if self.conf.distributed:
            self.conf.local_rank = int(os.environ['LOCAL_RANK'])
            os.environ['MASTER_ADDR'] = self.conf.dist_master_addr
            os.environ['MASTER_PORT'] = self.conf.dist_master_port
            dist.init_process_group(
                self.conf.dist_backend,
                init_method=self.conf.dist_init_method,
                rank=self.conf.local_rank,
                world_size=torch.cuda.device_count())
            os.environ['RANK'] = str(dist.get_rank())
            self.device = self.conf.local_rank
            self.log(
                f'local rank: {self.conf.local_rank} | '
                f'CUDA device: {self.device}')
            torch.cuda.set_device(self.device)
        else:
            self.conf.local_rank = 0

    def setup_model(self):
        """Create model and load model weights if specified.
        """
        self.model = self.make_model()
        self.maybe_load_model()
        self.model = self.model.to(self.device)
        self.log_model_params()

    def is_dist_main(self):
        return getattr(self.conf, 'local_rank', 0) == 0

    def distribute_model(self):
        if self.conf.distributed:
            self.log(f'Distributing model to device: {self.device}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                output_device=self.conf.local_rank,
                find_unused_parameters=True)

    def setup_early_stopping(self):
        if self.conf.early_stopping > 0:
            from ignite.handlers import EarlyStopping
            handler = EarlyStopping(
                patience=self.conf.early_stopping,
                score_function=self.checkpoint_score_function,
                trainer=self.train_engine,
                min_delta=1e-6)
            self.eval_engine.add_event_handler(Events.COMPLETED, handler)
            self.log(f'Early stopping patience: {self.conf.early_stopping}')

    @property
    def checkpoint_score_function(self):
        """The score function will be used to calculate the score to track for
        model checkpointing.
        """
        sign = {'max': 1, 'min': -1}[self.checkpoint_metric_optimum]

        def score_function(engine):
            metric_name = self.checkpoint_metric_name
            return sign * self.eval_engine.state.metrics[metric_name]
        return score_function

    @property
    def checkpoint_metric_name(self):
        """The metric to track for model checkpointing.
        """
        return 'acc'

    @property
    def checkpoint_metric_optimum(self):
        """Specifies whether high or low scores are good when tracking
        scores for model checkpointing.
        """
        return 'max'

    def make_model(self):
        raise NotImplementedError()

    def maybe_load_model(self):
        """Load model if a model_file is supplied.
        """
        if self.conf.model_file:
            self.log(f'loading model {self.conf.model_file}')
            state_dict = torch.load(self.conf.model_file, map_location='cpu')
            if self.conf.model_file.name.startswith('checkpoint'):
                state_dict = state_dict['model']
                state_dict = fix_dataparallel_statedict(self.model, state_dict)
            self.model.load_state_dict(state_dict)

    @property
    def last_checkpoint_file(self):
        """Returns the last model checkpoint found in the current run directory.
        """
        checkpoints = list(self.conf.rundir.glob('checkpoint_*.pt'))
        if checkpoints:
            last = sorted(checkpoints, key=os.path.getmtime)[-1]
            self.log(f'Found checkpoint: {last}')
            return last
        else:
            self.log('No checkpoint found.')
            return None

    def log_model_params(self):
        log_param_counts(self.model, self.log)

    def setup_optim(self):
        self.optim = self.make_optim()

    def setup_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(
            self.conf,
            self.optim,
            optimum=self.conf.lr_metric_optimum,
            n_train_steps=self.n_train_steps)

    def make_optim(self):
        return get_optim(
            self.conf,
            self.model,
            additional_params_dict=self.additional_params_dict)

    @property
    def additional_params_dict(self):
        """A dictionary from model parameters to optimizer configurations.
        This is useful when applying different learning rates to different
        parts of a model, e.g. a larger learning rate for a model head and
        a smaller learning rate for the body of a pretrained model.
        The dictionary will be passed to the optimzer as parameter group.
        """
        return None

    def load_data(self):
        raise NotImplementedError()

    @property
    def n_train_steps(self):
        """Computes the number of training steps, which is required for some
        learning rate schedulers.
        """
        return len(self.data.train_loader) * self.conf.max_epochs

    @cached_property
    def train_metrics(self):
        """Return the metrics to be calculated during training.
        """
        return self.metrics()

    @cached_property
    def eval_metrics(self):
        """Return the metrics to be calculated during evaluation.
        """
        return self.metrics(for_eval=True)

    @property
    def eval_event(self):
        return Events.EPOCH_COMPLETED

    @cached_property
    def train_engine(self):
        """The PyTorch Ignite engine which will perform the training loop
        and handle events.
        """
        engine = Engine(self.make_train_step())
        for metric_name, metric in self.train_metrics.items():
            metric.attach(engine, metric_name)

        if not hasattr(self, 'lr_scheduler') or self.conf.lr_scheduler == 'plateau':
            self.lr_scheduler_train_step = lambda: None
        else:
            if self.conf.lr_scheduler == 'warmup_linear':
                if self.is_dist_main():
                    self.log(f'n_train_steps: {self.n_train_steps}')
            self.lr_scheduler_train_step = self.lr_scheduler.step

        if self.is_dist_main():
            @engine.on(self.eval_event)
            def run_eval(_):
                self.log_results('train', engine.state.metrics)
                self.eval_engine.run(self.data.dev_loader)
                self.log_results('dev', self.eval_engine.state.metrics)
                self.log_metrics()

            if not self.conf.no_checkpoints:
                self.eval_engine.add_event_handler(
                    Events.COMPLETED, self.checkpointer, self.checkpoint_dict)
            self.eval_engine.add_event_handler(
                Events.COMPLETED, self.log_checkpoint)

            if self.conf.lr_scheduler == 'plateau':
                def plateau_step(_):
                    metrics = self.eval_engine.state.metrics
                    score = metrics[self.checkpoint_metric_name]
                    return self.lr_scheduler.step(score)

                self.eval_engine.add_event_handler(
                    Events.COMPLETED, plateau_step)

            if hasattr(self.data, 'test_loader'):
                @engine.on(Events.COMPLETED)
                def run_test(_):
                    self.test_engine.run(self.data.test_loader)
                    self.log_results('test', self.test_engine.state.metrics)

        return engine

    @cached_property
    def eval_engine(self):
        """The PyTorch Ignite engine which will perform an evaluation
        step during training and handle events.
        """
        engine = Engine(self.make_eval_step())
        for metric_name, metric in self.eval_metrics.items():
            metric.attach(engine, metric_name)
        return engine

    @cached_property
    def test_engine(self):
        """The PyTorch Ignite engine used for testing.
        """
        engine = Engine(self.make_eval_step())
        for metric_name, metric in self.eval_metrics.items():
            metric.attach(engine, metric_name)
        return engine

    def log_results(self, eval_name, metrics):
        metrics_str = ' | '.join(
            f'{metric} {val:.4f}' for metric, val in metrics.items())
        self.log(f"epoch {self.epoch:04d} {eval_name} | {metrics_str}")

    @property
    def actual_model(self):
        return getattr(self.model, 'module', self.model)

    @cached_property
    def checkpointer(self):
        return Checkpoint(
            to_save=self.checkpoint_dict,
            save_handler=DiskSaver(self.conf.rundir, require_empty=False),
            filename_prefix=self.checkpoint_prefix,
            score_name=self.checkpoint_metric_name,
            score_function=self.checkpoint_score_function,
            n_saved=3,
            )

    @property
    def checkpoint_prefix(self):
        return ''

    @property
    def checkpoint_dict(self):
        to_save = {
            'model': self.model,
            'optim': self.optim}
        if getattr(self, 'lr_scheduler', None):
            to_save['lr_scheduler'] = self.lr_scheduler
        return to_save

    def log_checkpoint(self, engine):
        self.log(self.checkpointer.last_checkpoint)

    @property
    def best_checkpoint(self):
        if not self.checkpointer._saved:
            return None
        return self.checkpointer._saved[-1][1]

    def make_train_step(self):
        """This method is the core of the training loop, also called "update step"
        in other frameworks.
        """
        scaler = torch.cuda.amp.GradScaler()

        def train_step(train_engine, batch):
            self.model.train()
            self.optim.zero_grad()
            batch = {k: v.to(device=self.device) for k, v in batch.items()}
            with autocast(enabled=not self.conf.no_fp16):
                result = self.model(batch)
                loss = result['loss'].mean()
            scaler.scale(loss).backward()
            clip_grad_norm_(
               self.model.parameters(), self.conf.max_grad_norm)
            scaler.step(self.optim)
            scaler.update()
            self.lr_scheduler_train_step()
            result = {
                k: v.detach().to(device='cpu') for k, v in result.items()}
            result['loss'] = loss.item()
            return result
        return train_step

    def make_eval_step(self):
        """This method is the core of the evaluation loop, also called
        "inference step" in other frameworks.
        """
        raise NotImplementedError()

    @property
    def epoch(self):
        """Return the current epoch.
        """
        return self.train_engine.state.epoch

    @property
    def exp_params(self):
        """Return all relevant parameters of the current experiment, e.g.
        for logging.
        """
        raise NotImplementedError

    def log_metrics(self):
        metrics = self.eval_engine.state.metrics
        self.exp_logger.log_metrics(metrics, step=self.epoch)

    def setup_bookkeeping(self):
        pass

    @property
    def event_handlers_train(self):
        """Some basic events that will be registered to run during training.
        """
        return [
            (Events.STARTED, self.load_state),
            (Events.EPOCH_COMPLETED, self.save_state),
            (Events.COMPLETED, self.save_results)]

    def load_state(self):
        objs_and_state_files = [
            (self.train_engine.state, self.conf.trainer_state_file),
            (self.checkpointer, self.conf.checkpointer_state_file)]
        for obj, state_file in objs_and_state_files:
            if state_file.exists():
                self.log(f'loading {state_file}')
                if state_file.suffix == '.pt':
                    obj.load_state_dict(torch.load(state_file))
                else:
                    state = json_load(state_file)
                    for k, v in state.items():
                        setattr(obj, k, v)
                        self.log(f'{k}: {getattr(obj, k)}')
        checkpoint_file = self.last_checkpoint_file
        if checkpoint_file:
            checkpoint = torch.load(checkpoint_file)
            for k, v in checkpoint.items():
                if k == 'model':
                    v = fix_dataparallel_statedict(self.model, v)
                getattr(self, k).load_state_dict(v)
                self.log(f'loaded state dict: {k}')

    def save_state(self, engine=None):
        trainer_state = dict(
            epoch=engine.state.epoch,
            iteration=engine.state.iteration,
            metrics=engine.state.metrics
            )
        json_dump(trainer_state, self.conf.trainer_state_file)
        torch.save(
            self.checkpointer.state_dict(), self.conf.checkpointer_state_file)

    def save_results(self, engine=None):
        checkpoint_file = self.best_checkpoint or 'no_checkpoint'
        params = self.exp_logger.exp_params
        run_info = dict(
            checkpoint_file=checkpoint_file,
            final_epoch=self.train_engine.state.epoch)
        self.exp_logger.log_params(run_info)
        self.exp_logger.log_artifacts()
        metrics = self.eval_engine.state.metrics
        self.results = dict(**params, **run_info, **metrics)
        print(self.results)
        if not self.conf.dev:
            fname = (self.conf.runid or 'results') + '.json'
            results_file = mkdir(self.conf.results_dir) / fname
            json_dump(self.results, results_file)
            self.log(results_file)

    def save_stdout(self):
        jobid = getattr(self.conf, 'jobid', None)
        if jobid:
            stdout_file = (self.conf.stdout_dir / str(jobid)).expanduser()
            stdout_copy = self.conf.rundir / 'stdout.txt'
            try:
                import shutil
                shutil.copy(str(stdout_file), str(stdout_copy))
            except Exception:
                pass

    def cleanup(self):
        if self.conf.distributed and self.is_dist_main():
            # Pytorch distributed processes continue running after the main
            # process has finished, which means the cluster node we're running
            # on doesn't get freed up.
            # try to kill those manually, might kill other python processes
            # if we're not running exclusively ¯\_(ツ)_/¯
            dist.destroy_process_group()
            main_pid = os.getpid()
            output = subprocess.check_output("pidof -c python".split())
            pids = list(map(int, output.split()))
            for pid in pids:
                if pid > main_pid:
                    os.kill(pid, 9)

    def train(self):
        """Do a complete model training run.
        """
        self.start_run()
        self.train_engine.run(
            self.data.train_loader, max_epochs=self.conf.max_epochs)
        self.end_run()
        if self.is_dist_main():
            return self.results

    def test(self):
        """Run a trained model on the test set of specified dataset,
        but do not perform any traiing."""
        self.start_run()
        self.eval_engine.run(self.data.test_loader)
        self.log_metrics()
        self.log_results('test', self.eval_engine.state.metrics)
        self.save_results()
        self.end_run()
        if self.is_dist_main():
            return self.results

    def cache_dataset(self):
        """Cache the specified dataset. If loading and tensorizing a large
        takes a long time, it is sometimes useful to only do the cachin on
        cheap CPU server, and the load the cached file when training on a GPU
        node.
        """
        self.load_data()

    def start_run(self):
        if self.is_dist_main():
            self.exp_logger.start_run()

    def end_run(self):
        if self.is_dist_main():
            self.save_stdout()
            self.exp_logger.end_run()
        self.cleanup()
