import datetime
import hashlib
import json
import os
import random
import shutil
import warnings
import urllib.request
from tqdm import tqdm
import mlflow
import wandb
import numpy as np
import torch, time

from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from ticl.model_configs import get_model_default_config
from ticl.config_utils import flatten_dict
import itertools

from ticl.rl_validation import evaluate_rlpfn_on_gym_envs

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def fetch_model(file_name):
    model_path = Path(get_module_path()) / 'models_diff' / file_name
    if not model_path.exists():
        url = f'https://amuellermothernet.blob.core.windows.net/models/{file_name}'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading model from {url} to {model_path}. This can take a bit.")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=model_path, reporthook=t.update_to)
    return model_path.resolve()


def get_module_path():
    return Path(__file__).parent.resolve()


class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'


def get_nan_value(set_value_to_nan=0.0):
    if random.random() < set_value_to_nan:
        return float('nan')
    else:
        return random.choice([-999, 0, 1, 999])
    

def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.-num/x.shape[dim]
    return value / num


def torch_masked_std(x, mask, dim=0):
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))


def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)


def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)


def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data


def remove_outliers(X, n_sigma=4, normalize_positions=-1, categorical_features=None):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    if categorical_features:
        categorical_mask = torch.zeros(X.shape[2], dtype=torch.bool, device=X.device)
        categorical_mask.scatter_(0, torch.tensor(categorical_features, device=X.device, dtype=int), 1.)

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    if categorical_features:
        X = torch.where(categorical_mask, X, torch.maximum(-torch.log(1+torch.abs(X)) + lower, X))
        X = torch.where(categorical_mask, X, torch.minimum(torch.log(1+torch.abs(X)) + upper, X))
    else:
        X = torch.maximum(-torch.log(1+torch.abs(X)) + lower, X)
        X = torch.minimum(torch.log(1+torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X


def print_on_master_only(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(device):
    # print('init dist')
    if 'LOCAL_RANK' in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print('torch.distributed.launch and my rank is', rank)
        torch.cuda.set_device(rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=20),
                                             world_size=torch.cuda.device_count(), rank=rank)
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
              "only I can print, but when using print(..., force=True) it will print on all ranks.")
        return True, rank, f'cuda:{rank}'
    elif 'SLURM_PROCID' in os.environ and torch.cuda.device_count() > 1:
        # this is for multi gpu when starting with submitit
        assert device != 'cpu:0'
        rank = int(os.environ['SLURM_PROCID'])
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        print('distributed submitit launch and my rank is', rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=20),
                                             world_size=torch.cuda.device_count(), rank=rank)
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
              "only I can print, but when using print(..., force=True) it will print on all ranks.")

        return True, rank, f'cuda:{rank}'
    else:
        # print('Not using distributed')
        # will not change any of the behavior of print, but allows putting the force=True in the print calls
        print_on_master_only(True)
        return False, 0, device

# NOP function for python with statements (x = NOP(); with x:)


class NOP():
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def check_compatibility(dl):
    if hasattr(dl, 'num_outputs'):
        print('`num_outputs` for the DataLoader is deprecated. It is assumed to be 1 from now on.')
        assert dl.num_outputs != 1, "We assume num_outputs to be 1. Instead of the num_ouputs change your loss." \
                                    "We specify the number of classes in the CE loss."


def normalize_by_used_features_f(x, num_features_used, num_features):
    return x / (num_features_used / num_features)


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, min_lr=None, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


class ReduceLROnSpike:
    """Reduce learning rate when a metric has bounced up.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        smoothing (int): Number of epochs with over which to smooth recent performance.
            Default: 10.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        tolerance (int): Multiple of std from recent data to be considered a spike.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, smoothing=10,
                 min_lr=0, verbose=False, tolerance=4, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.smoothing = smoothing
        self.verbose = verbose
        self.mode = mode
        self.eps = eps
        self.tolerance = tolerance
        self.last_epoch = 0
        self.recent_losses = []
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if len(self.recent_losses) < self.smoothing:
            self.recent_losses.append(current)
        else:
            sign = -1 if self.mode == 'min' else 1

            if np.mean(self.recent_losses) < current + self.tolerance * sign * np.std(self.recent_losses):
                if self.verbose:
                    print("That loss looks bad!")
                    print("Recent losses:", self.recent_losses)
                    print("Current loss:", current)
                self._reduce_lr(epoch)
                self.recent_losses = []
            else:
                self.recent_losses = self.recent_losses[1:] + [current]

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print(f'Epoch {epoch_str}: reducing learning rate of group {i} from {old_lr:.4e} to {new_lr:.4e}.')

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr


def init_device(gpu_id, use_cpu):
    # Single GPU training, get GPU ID from command line
    if 'LOCAL_RANK' in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print('torch.distributed.launch and my rank is', rank)
        num_gpus = int(os.environ["WORLD_SIZE"])
        raise ValueError("Gave up on multi-gpu for now")
    else:
        rank = 0
        num_gpus = 1

    device = "cuda"
    if gpu_id is not None:
        if use_cpu:
            raise ValueError("Can't use cpu and gpu at the same time")
        device = f'cuda:{gpu_id}'
    elif use_cpu:
        device = 'cpu'
    return device, rank, num_gpus


def _resolve_max_filename_component_bytes(default_value=255):
    try:
        v = int(os.environ.get("TICL_MAX_FILENAME_COMPONENT_BYTES", str(default_value)))
    except Exception:
        v = int(default_value)
    # Common Linux filesystems (ext4/xfs) cap filename component at 255 bytes.
    return int(min(255, max(32, v)))


def _truncate_component_with_hash(name, max_bytes):
    name = str(name)
    raw = name.encode("utf-8")
    if len(raw) <= int(max_bytes):
        return name
    digest = hashlib.sha1(raw).hexdigest()[:10]
    suffix = f"_h{digest}"
    budget = int(max_bytes) - len(suffix.encode("utf-8"))
    if budget <= 0:
        return suffix.encode("utf-8")[: int(max_bytes)].decode("utf-8", errors="ignore")
    prefix_bytes = raw[:budget]
    while True:
        try:
            prefix = prefix_bytes.decode("utf-8")
            break
        except UnicodeDecodeError:
            prefix_bytes = prefix_bytes[:-1]
            if len(prefix_bytes) == 0:
                prefix = ""
                break
    prefix = prefix.rstrip("._-")
    if not prefix:
        return suffix
    return prefix + suffix


def enforce_path_filename_limit(path, *, max_component_bytes=None):
    """
    Ensure path basename fits filesystem filename-component limits.
    Preserves extension and injects a short hash when truncation is needed.
    """
    if path is None:
        return None
    path_str = str(path)
    if not path_str.strip():
        return path_str
    if max_component_bytes is None:
        max_component_bytes = _resolve_max_filename_component_bytes()
    else:
        max_component_bytes = int(min(255, max(32, int(max_component_bytes))))

    parent = os.path.dirname(path_str)
    base = os.path.basename(path_str)
    stem, ext = os.path.splitext(base)
    ext_bytes = len(ext.encode("utf-8"))
    stem_budget = max(8, max_component_bytes - ext_bytes)
    safe_stem = _truncate_component_with_hash(stem, stem_budget)
    safe_base = f"{safe_stem}{ext}"
    if parent:
        return os.path.join(parent, safe_base)
    return safe_base


_MODEL_STRING_EPHEMERAL_KEYS = {
    "pg_phase_log_file",
    "train_profiler_output_path",
    "train_gpu_observer_output_path",
    "train_gpu_stage_output_path",
    "train_kernel_profiler_output_dir",
    "pg_compile_observe_output_path",
}


def get_model_string(config, num_gpus, device, parser):
    # get the subparser for the model type
    subparser = parser._actions[1].choices[config['model_type']]
    config_shorthands = {arg.dest: arg.option_strings[0].replace('-', '') for arg in subparser._actions if arg.option_strings}
    mm = config['model_type']
    model_type_string = 'mn' if mm in ["mlp", "mothernet"] else mm
    default_config_flat = flatten_dict(get_model_default_config(config['model_type']), only_last=True)
    config_flat = flatten_dict({k: v for k, v in config.items() if k != 'orchestration'}, only_last=True)
    config_string = ""
    for k in sorted(config_flat.keys()):
        if k in ['run_id', 'use_cpu', 'gpu_id', 'help', 'model_type', 'num_gpus', 'device', 'nhead']:
            continue
        if k in _MODEL_STRING_EPHEMERAL_KEYS:
            continue
        v = config_flat[k]
        if k not in default_config_flat:
            print(f"Warning: {k} not in default config")
            continue
        if v != default_config_flat[k]:
            shortname = config_shorthands.get(k, k)
            if isinstance(v, float):
                config_string += f"_{shortname}{v:.4g}"
            else:
                config_string += f"_{shortname}{v}"
    gpu_string = f"_{num_gpus}_gpu{'s' if num_gpus > 1 else ''}" if device != 'cpu' else '_cpu'
    if gpu_string == "_1_gpu":
        gpu_string = ""
    model_string = (
        f"{model_type_string}{config_string}{gpu_string}"
        f"{'_continue' if config['orchestration']['continue_run'] else '_warm' if config['orchestration']['warm_start_from'] else ''}"
    )
    model_string = model_string + '_' + datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    # Reserve bytes for suffixes used by downstream files, e.g.
    # "{model_string}_epoch_{epoch}.cpkt" and "{model_string}.log".
    max_component = _resolve_max_filename_component_bytes()
    reserved_suffix = len("_epoch_999999999.cpkt".encode("utf-8"))
    model_string_budget = max(32, int(max_component) - int(reserved_suffix))
    model_string_safe = _truncate_component_with_hash(model_string, model_string_budget)
    if model_string_safe != model_string:
        print(
            "[filename-limit] model_string was truncated to fit filesystem limits "
            f"(budget={model_string_budget} bytes)."
        )
    model_string = model_string_safe
    if config['orchestration']['st_checkpoint_dir'] is not None:
        with open(f"{config['orchestration']['st_checkpoint_dir']}/model_string.txt", 'w') as f:
            f.write(model_string)
    return model_string


def _load_checkpoint_metric_map(path):
    if not os.path.exists(path):
        return {"metric": None, "mode": None, "scores": {}}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"metric": None, "mode": None, "scores": {}}
        scores = data.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}
        return {
            "metric": data.get("metric"),
            "mode": data.get("mode"),
            "scores": {str(k): float(v) for k, v in scores.items()},
        }
    except Exception:
        return {"metric": None, "mode": None, "scores": {}}


def _save_checkpoint_metric_map(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _checkpoint_is_better(mode, lhs, rhs):
    if mode == "max":
        return float(lhs) > float(rhs)
    return float(lhs) < float(rhs)


def _sanitize_metric_token(token):
    token = str(token).strip()
    if not token:
        return "unknown"
    out = []
    for ch in token:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _flatten_named_scores(score_obj, prefix=""):
    if isinstance(score_obj, dict):
        flat = []
        for key in sorted(score_obj.keys(), key=lambda k: str(k)):
            key_token = _sanitize_metric_token(key)
            next_prefix = key_token if not prefix else f"{prefix}_{key_token}"
            flat.extend(_flatten_named_scores(score_obj[key], next_prefix))
        return flat
    label = prefix if prefix else "value"
    return [(label, score_obj)]


def _format_log_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if np.isnan(fv):
            return "nan"
        if np.isposinf(fv):
            return "inf"
        if np.isneginf(fv):
            return "-inf"
        return f"{fv:.16g}"
    return str(value)


def make_training_callback(
    save_every, 
    model_string, 
    base_path, 
    report, 
    config, 
    use_mlflow, 
    checkpoint_dir, 
    classification, 
    validate
):
    from ticl.model_builder import save_model
    config = config.copy()

    def save_callback(model, optimizer, scheduler, epoch):
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        log_file = f'{base_path}/log/{model_string}.log'
        if epoch == "start":
            print(f"Starting training of model {model_string}")
            return
        try:
            os.makedirs(f"{base_path}/log", exist_ok=True)
            with open(log_file, 'a') as f:
                if config.get("model_type") == "rlpfn":
                    f.write(f'Epoch {epoch} train_loss {model.losses[-1]} learning_rate {model.learning_rates[-1]}\n')
                    pg_diag = getattr(model, "last_pg_epoch_metrics", None)
                    if isinstance(pg_diag, dict) and len(pg_diag) > 0:
                        diag_fields = []
                        for k in sorted(pg_diag.keys()):
                            v = pg_diag.get(k, None)
                            if v is None:
                                continue
                            if isinstance(v, (bool, np.bool_)):
                                diag_fields.append(f"{k}={int(v)}")
                            elif isinstance(v, (int, np.integer)):
                                diag_fields.append(f"{k}={int(v)}")
                            elif isinstance(v, (float, np.floating)):
                                fv = float(v)
                                if np.isfinite(fv):
                                    diag_fields.append(f"{k}={fv:.6g}")
                            else:
                                diag_fields.append(f"{k}={str(v)}")
                        if diag_fields:
                            f.write(f"Epoch {epoch} pg_diag {' '.join(diag_fields)}\n")
                    ppo_diag = getattr(model, "last_ppo_epoch_metrics", None)
                    if isinstance(ppo_diag, dict) and len(ppo_diag) > 0:
                        diag_fields = []
                        for k in sorted(ppo_diag.keys()):
                            v = ppo_diag.get(k, None)
                            if v is None:
                                continue
                            if isinstance(v, (bool, np.bool_)):
                                diag_fields.append(f"{k}={int(v)}")
                            elif isinstance(v, (int, np.integer)):
                                diag_fields.append(f"{k}={int(v)}")
                            elif isinstance(v, (float, np.floating)):
                                fv = float(v)
                                if np.isfinite(fv):
                                    diag_fields.append(f"{k}={fv:.6g}")
                            else:
                                diag_fields.append(f"{k}={str(v)}")
                        if diag_fields:
                            f.write(f"Epoch {epoch} ppo_diag {' '.join(diag_fields)}\n")
                    ppo_logger_metrics = getattr(model, "last_ppo_logger_metrics", None)
                    if isinstance(ppo_logger_metrics, dict) and len(ppo_logger_metrics) > 0:
                        def _ppo_metric_sort_key(key):
                            key_str = str(key)
                            if key_str.startswith("rollout/"):
                                return (0, key_str)
                            if key_str.startswith("time/"):
                                return (1, key_str)
                            if key_str.startswith("train/"):
                                return (2, key_str)
                            return (3, key_str)

                        for key in sorted(ppo_logger_metrics.keys(), key=_ppo_metric_sort_key):
                            value = ppo_logger_metrics.get(key, None)
                            if value is None:
                                continue
                            f.write(
                                f"Epoch {epoch} ppo_metric {key} {_format_log_scalar(value)}\n"
                            )
                else:
                    f.write(f'Epoch {epoch} loss {model.losses[-1]} learning_rate {model.learning_rates[-1]}\n')
        except Exception as e:
            print(f'Failed to write to log file {log_file}: {e}')

        if epoch != "on_exit":
            wallclock_ticker = max(1, int(model.wallclock_times[-1]//(60 * 5)))
            if use_mlflow:
                mlflow.log_metric(key="wallclock_time", value=model.wallclock_times[-1], step=epoch)
                mlflow.log_metric(key="loss", value=model.losses[-1], step=epoch)
                mlflow.log_metric(key="learning_rate", value=model.learning_rates[-1], step=epoch)
                mlflow.log_metric(key="wallclock_ticker", value=wallclock_ticker, step=epoch)
                mlflow.log_metric(key="epoch", value=epoch, step=epoch)
            if wandb.run is not None:
                wandb.log({"loss": model.losses[-1], "learning_rate": model.learning_rates[-1], "wallclock_time": model.wallclock_times[-1],
                           "wallclock_ticker": wallclock_ticker, "epoch": epoch})
            if report is not None:
                # synetune callback
                if config.get("model_type") == "rlpfn":
                    report(epoch=epoch, train_loss=model.losses[-1], wallclock_time=wallclock_ticker)
                else:
                    report(epoch=epoch, loss=model.losses[-1], wallclock_time=wallclock_ticker)  # every 5 minutes

        if (epoch == "on_exit") or epoch % save_every == 0:
            if checkpoint_dir is not None:
                if epoch == "on_exit":
                    return
                file_name = f'{base_path}/checkpoint.mothernet'
            else:
                file_name = f'{base_path}/models_diff/{model_string}_epoch_{epoch}.cpkt'
            try:
                os.makedirs(f"{base_path}/models_diff", exist_ok=True)
                disk_usage = shutil.disk_usage(f"{base_path}/models_diff")
                if disk_usage.free < 1024 * 1024 * 1024 * 2:
                    print("Not saving model, not enough disk space")
                    print("DISK FULLLLLLL")
                    return
                with open(log_file, 'a') as f:
                    f.write(f'Saving model to {file_name}\n')
                print(f'Saving model to {file_name}')
                config['epoch_in_training'] = epoch
                config['learning_rates'] = model.learning_rates
                config['losses'] = model.losses
                config['wallclock_times'] = model.wallclock_times

                save_model(model, optimizer, scheduler, base_path, file_name, config)
            
            except Exception as e:
                print("WRITING TO MODEL FILE FAILED")
                print(e)

            on_cuda = next(model.parameters()).is_cuda

            if epoch != "on_exit":
                inference_time = None
                validation_score = None
                per_dataset_score = {}
                if on_cuda:
                    gpu_start_time = torch.cuda.Event(enable_timing=True)
                    gpu_end_time = torch.cuda.Event(enable_timing=True)
                if validate:
                    inference_start = time.time()
                    if on_cuda:
                        gpu_start_time.record()
                    
                    validation_score, per_dataset_score = validate_model(model, config)
                    
                    inference_end = time.time()
                    if on_cuda:
                        gpu_end_time.record()
                        torch.cuda.synchronize()
                    
                    inference_time = inference_end - inference_start
                    if on_cuda:
                        gpu_inference_time = gpu_start_time.elapsed_time(gpu_end_time)
                    else:
                        gpu_inference_time = 0

                    if config.get("model_type") == "rlpfn":
                        print(f"[ext-rl] mean_return_all={validation_score}")
                        per_env_returns = _flatten_named_scores(per_dataset_score)
                        for env_name, env_return in per_env_returns:
                            print(f"[ext-rl] return_{env_name}={_format_log_scalar(env_return)}")
                        try:
                            with open(log_file, "a") as f:
                                f.write(f"Epoch {epoch} mean_return_all {validation_score}\n")
                                for env_name, env_return in per_env_returns:
                                    f.write(
                                        f"Epoch {epoch} return_{env_name} "
                                        f"{_format_log_scalar(env_return)}\n"
                                    )
                        except Exception:
                            pass
                    else:
                        print(f"Validation score: {validation_score}")

                    if use_mlflow:
                        if np.isfinite(float(validation_score)):
                            if config.get("model_type") == "rlpfn":
                                mlflow.log_metric(key="mean_return_all", value=float(validation_score), step=epoch)
                            mlflow.log_metric(key="val_score", value=float(validation_score), step=epoch)
                        for dataset, score in per_dataset_score.items():
                            if isinstance(score, dict):
                                for sub_key, sub_val in score.items():
                                    if isinstance(sub_val, (int, float)) and np.isfinite(float(sub_val)):
                                        mlflow.log_metric(
                                            key=f"val_score_{dataset}_{sub_key}",
                                            value=float(sub_val),
                                            step=epoch,
                                        )
                            elif isinstance(score, (int, float)) and np.isfinite(float(score)):
                                mlflow.log_metric(key=f"val_score_{dataset}", value=float(score), step=epoch)
                            
                    if wandb.run is not None:
                        val_metrics = {
                            "val_score": validation_score, 
                            "epoch": epoch, 
                            'inference_time': inference_time,
                            'gpu_inference_time': gpu_inference_time,
                        }
                        if config.get("model_type") == "rlpfn":
                            val_metrics["mean_return_all"] = validation_score
                        
                        for dataset, score in per_dataset_score.items():
                            if isinstance(score, dict):
                                for sub_key, sub_val in score.items():
                                    val_metrics[f"val_score_{dataset}_{sub_key}"] = sub_val
                            else:
                                val_metrics[f"val_score_{dataset}"] = score
                        wandb.log(val_metrics)

                metric_name = "loss"
                metric_mode = "min"
                metric_value = float(model.losses[-1])
                if (
                    config.get("model_type") == "rlpfn"
                    and validate
                    and validation_score is not None
                    and np.isfinite(float(validation_score))
                ):
                    metric_name = "mean_return"
                    metric_mode = "max"
                    metric_value = float(validation_score)

                if checkpoint_dir is None and os.path.exists(file_name):
                    metric_map_path = f"{base_path}/models_diff/checkpoint_metrics.json"
                    metric_map = _load_checkpoint_metric_map(metric_map_path)
                    if metric_map.get("metric") != metric_name or metric_map.get("mode") != metric_mode:
                        metric_map = {"metric": metric_name, "mode": metric_mode, "scores": {}}

                    # Keep only entries that still exist on disk (resume-safe).
                    cleaned_scores = {
                        ckpt: score
                        for ckpt, score in metric_map.get("scores", {}).items()
                        if os.path.exists(ckpt)
                    }
                    metric_map = {"metric": metric_name, "mode": metric_mode, "scores": cleaned_scores}
                    metric_map["scores"][file_name] = metric_value

                    to_remove = []
                    for old_ckpt, old_score in metric_map["scores"].items():
                        if old_ckpt == file_name:
                            continue
                        if _checkpoint_is_better(metric_mode, metric_value, old_score):
                            to_remove.append(old_ckpt)

                    for old_ckpt in to_remove:
                        try:
                            if os.path.exists(old_ckpt):
                                print(
                                    f"Removing old checkpoint {old_ckpt} "
                                    f"({metric_name}={metric_map['scores'][old_ckpt]:.6f})"
                                )
                                os.remove(old_ckpt)
                        except Exception as e:
                            print(f"Failed to remove old checkpoint {old_ckpt}: {e}")
                        metric_map["scores"].pop(old_ckpt, None)

                    _save_checkpoint_metric_map(metric_map_path, metric_map)
                    try:
                        with open(log_file, "a") as f:
                            f.write(
                                f"checkpoint_metric epoch={epoch} name={metric_name} mode={metric_mode} "
                                f"value={metric_value:.6f}\n"
                            )
                    except Exception:
                        pass
                if validate: 
                    if config.get("model_type") == "rlpfn":
                        return {
                            "inference_time": inference_time,
                            "mean_return_all": validation_score,
                        }
                    return inference_time

    return save_callback


def synetune_handle_checkpoint(args):
    # handle syne-tune restarts
    checkpoint_dir = args.st_checkpoint_dir
    base_path = args.base_path
    warm_start_from = args.warm_start_from
    continue_run = args.continue_run
    report = None
    if checkpoint_dir is not None:
        from syne_tune import Reporter
        report = Reporter()
        base_path = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.mothernet"
        if checkpoint_path.exists():
            continue_run = True
            warm_start_from = checkpoint_path
    return base_path, continue_run, warm_start_from, report


def get_init_method(init_method):
    if init_method is None:
        return None
    if init_method == "kaiming-uniform":
        method = nn.init.kaiming_uniform_
    if init_method == "kaiming-normal":
        method = nn.init.kaiming_normal_
    if init_method == "xavier-uniform":
        method = nn.init.xavier_uniform_
    if init_method == "xavier-normal":
        method = nn.init.xavier_normal_

    def init_weights_inner(layer):
        if isinstance(layer, nn.Linear):
            method(layer.weight)
            nn.init.zeros_(layer.bias)
    return init_weights_inner


def validate_model(model, config):
    if config.get("model_type") == "rlpfn":
        enabled = bool(config.get("orchestration", {}).get("rl_validate_enabled", True))
        if not enabled:
            return float("nan"), {}
        return evaluate_rlpfn_on_gym_envs(model=model, config=config)

    from ticl.datasets import load_openml_list, open_cc_valid_dids, open_cc_valid_dids_regression, open_cc_large_dids, new_valid_dids

    from ticl.models.gamformer import GAMformer
    from ticl.models.mothernet_additive import MotherNetAdditive
    from ticl.models.mothernet import MotherNet
    from ticl.models.la_mothernet import SSMMotherNet
    from ticl.models.tabflex import TabFlex
    from ticl.models.tabpfn import TabPFN
    from ticl.models.perceiver import TabPerceiver
    from ticl.models.biattention_tabpfn import BiAttentionTabPFN
    from ticl.prediction import GAMformerClassifier, MotherNetClassifier, TabPFNClassifier, GAMformerRegressor 
    from ticl.evaluation.tabular_evaluation import eval_on_datasets
    from ticl.evaluation import tabular_metrics
    from uuid import uuid4

    if 'transformer' in config:
        attention_type = 'transformer'
    elif 'linear_attention' in config:
        attention_type = 'linear_attention'
    else:
        raise ValueError(f"Unknown attention type")
    
    if config[attention_type]['classification_task'] or config['openmlloader']['valid_data'] in ['large', 'new']:
        if config['openmlloader']['valid_data'] == 'new':
            open_cc_dids = new_valid_dids
        elif config['openmlloader']['valid_data'] == 'large':
            open_cc_dids = open_cc_large_dids
        else:
            open_cc_dids = open_cc_valid_dids
        cc_valid_datasets_multiclass, _ = load_openml_list(
            open_cc_dids, 
            multiclass=True, 
            shuffled=True, 
            filter_for_nan=False, 
            max_samples=1000000,
            num_feats=5000, 
            max_num_classes=100,
            return_capped=True, 
            classification=True,
        )

        if isinstance(model, (GAMformer, MotherNetAdditive)):
            clf = GAMformerClassifier(
                device=config['device'], 
                model=model, 
                config=config
            )
        elif isinstance(model, (MotherNet, SSMMotherNet)):
            clf = MotherNetClassifier(
                device=config['device'], 
                model=model, 
                config=config
            )
        elif isinstance(model, (TabPFN, BiAttentionTabPFN, TabFlex)):
            clf = TabPFNClassifier(
                device=config['device'], 
                model=model, 
                config=config, 
                N_ensemble_configurations=1
            )
        else:
            raise ValueError(f"Model {model} not supported for validation")
        base_path = 'models_diff/validation'
        results = eval_on_datasets(
            'multiclass', 
            clf, 
            f"valid_run_{uuid4()}", 
            cc_valid_datasets_multiclass,
            metric_used=tabular_metrics.auc_metric, 
            split_numbers=[1, 2, 3, 4, 5], 
            eval_positions=[None],
            max_times=[1], 
            n_samples=config['openmlloader']['max_samples'], 
            base_path=base_path, 
            overwrite=False, 
            n_jobs=1, 
            device=config['device'],
            save=False,
            max_features=config['prior']['num_features'],
            pca = config['openmlloader']['pca'],
        )
        mean_auc = np.array([r['mean_metric'] for r in results]).mean()
        # maybe pandas would be easier lol?
        per_dataset_scores = {key: np.mean([g['mean_metric'] for g in group]) for key, group in itertools.groupby(results, lambda x: x['dataset'])}
        return mean_auc, per_dataset_scores
    else:
        # regression
        cc_valid_datasets_regression, _ = load_openml_list(
            open_cc_valid_dids_regression, multiclass=False, shuffled=True, filter_for_nan=False, max_samples=10000,
            num_feats=100, return_capped=False, classification=False)

        if isinstance(model, (GAMformer, MotherNetAdditive)):
            clf = GAMformerRegressor(device=config['device'], model=model, config=config)
        else:
            raise ValueError(f"Model {model} not supported for validation")
        base_path = 'models_diff/validation'
        results = eval_on_datasets(
            'regression', 
            clf, 
            f"valid_run_{uuid4()}", 
            cc_valid_datasets_regression,
            metric_used=tabular_metrics.root_mean_squared_error_metric, 
            split_numbers=[1, 2, 3, 4, 5],
            eval_positions=[1000], 
            max_times=[1], 
            n_samples=2000, 
            base_path=base_path,
            overwrite=False, 
            n_jobs=1, 
            device=config['device'], 
            save=False, 
            max_features=config['prior']['num_features'],
            pca = config['openmlloader']['pca'],
        )
        mean_auc = np.array([r['mean_metric'] for r in results]).mean()
        # maybe pandas would be easier lol?
        per_dataset_scores = {key: np.mean([g['mean_metric'] for g in group]) for key, group in itertools.groupby(results, lambda x: x['dataset'])}
        return mean_auc, per_dataset_scores
