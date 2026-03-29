import os, pdb
import subprocess as sp

import torch, wandb
from torch import nn

import ticl.models.encoders as encoders
from ticl.dataloader import get_dataloader
from ticl.train import train
from ticl.model_configs import get_model_default_config
from ticl.models.mothernet_additive import MotherNetAdditive
from ticl.models.perceiver import TabPerceiver
from ticl.models.rwkv7_pfn import RWKV7PFN
from ticl.models.tabpfn import TabPFN
from ticl.models.biattention_tabpfn import BiAttentionTabPFN
from ticl.models.gamformer import GAMformer
from ticl.models.mothernet import MotherNet
from ticl.config_utils import nested_dict

try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)


def get_criterion(max_num_classes):
    if max_num_classes == 0:
        loss = nn.MSELoss(reduction='none')
    elif max_num_classes == 2:
        loss = nn.BCEWithLogitsLoss(reduction='none')
    elif max_num_classes > 2:
        loss = nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError(f"Invalid number of classes: {max_num_classes}")
    return loss


def save_model(model, optimizer, scheduler, path, filename, config_sample):
    optimizer_dict = optimizer.state_dict() if optimizer is not None else None
    extra_state = {}
    for candidate in (model, getattr(model, "module", None), getattr(model, "_orig_mod", None)):
        if candidate is None:
            continue
        ppo_policy_state = candidate.__dict__.get("_validation_ppo_policy_state", None)
        if isinstance(ppo_policy_state, dict):
            extra_state["validation_ppo_policy_state"] = {
                str(key): value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in ppo_policy_state.items()
            }
            break

    import cloudpickle
    torch.save(
        (model.state_dict(), optimizer_dict, scheduler, config_sample, extra_state),
        os.path.join(path, filename),
        pickle_module=cloudpickle,
    )


def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode('ascii')
    return memory_free_info


@cache
def load_model(path, device, verbose=False):
    states = torch.load(path, map_location='cpu', weights_only=False)
    model_state = states[0]
    if isinstance(states, (list, tuple)) and len(states) >= 5 and isinstance(states[4], dict):
        config_sample = states[3]
        extra_state = states[4]
    else:
        config_sample = states[-1]
        extra_state = {}
    if 'y_encoder' not in config_sample and 'onehot' in str(path):
        # workaround for the single model that was saved without y_encoder
        # that happens to be my reference model.
        config_sample['y_encoder'] = 'one_hot'
    _, model, *_ = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model_state.pop("criterion.weight", None)

    decoder_summary_weights = ["query", "output_layer.q_proj_weight", "output_layer.in_proj_weight", "output_layer.k_proj_weight", "output_layer.v_proj_weight",
                               "output_layer.in_proj_bias", "output_layer.out_proj.weight", "output_layer.out_proj.bias"]
    for weights in decoder_summary_weights:
        full_name = "decoder." + weights
        if full_name in model_state:
            model_state['decoder.summary_layer.' + weights] = model_state.pop(full_name)

    if "encoder.weight" in model_state and "model_type" in config_sample and config_sample['model_type'] == "additive":
        model_state['encoder.1.weight'] = model_state.pop("encoder.weight")
        model_state['encoder.1.bias'] = model_state.pop("encoder.bias")

    for module_name in list(model_state.keys()):
        if module_name.startswith("ssm"):
            # for backwards compatibility
            model_state["linear_attention" + module_name[3:]] = model_state.pop(module_name)

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    validation_ppo_policy_state = extra_state.get("validation_ppo_policy_state", None) if isinstance(extra_state, dict) else None
    if isinstance(validation_ppo_policy_state, dict):
        model.__dict__["_validation_ppo_policy_state"] = validation_ppo_policy_state

    return model, config_sample


def get_y_encoder(config, attention_type):
    if config[attention_type]['y_encoder'] == 'one_hot':
        y_encoder = encoders.OneHotAndLinear(config['prior']['classification']['max_num_classes'], emsize=config[attention_type]['emsize'])
    elif config[attention_type]['y_encoder'] == 'linear':
        y_encoder = encoders.Linear(1, emsize=config[attention_type]['emsize'])
    elif config[attention_type]['y_encoder'] in ['none', 'None', None]:
        y_encoder = None
    else:
        raise ValueError(f"Unknown y_encoder: {config[attention_type]['y_encoder']}")
    return y_encoder


def old_config_to_new(old_config, new_config):
    # this is not for restarting learning, only inference, so it doesn't convert orchestration parameters
    # it takes a flat config and converts it to a nested one based on the structure of new_config
    old_config['learning_rate'] = old_config.pop('lr')
    if "bptt" in old_config:
        old_config['n_samples'] = old_config.pop('bptt')
    old_config.update(old_config.pop("differentiable_hyperparameters", {}))
    if "y_encoder" not in old_config:
        old_config['y_encoder'] = 'linear'
    if "decoder_em_size" in old_config:
        old_config['decoder_embed_dim'] = old_config.pop('decoder_em_size')
    if "model_maker" in old_config:
        old_config['model_type'] = old_config.pop('model_maker')
    if "em_size" in old_config:
        old_config['emsize'] = old_config.pop('em_size')
    if "aggregate_gradients" in old_config:
        old_config['aggregate_k_gradients'] = old_config.pop('aggregate_gradients')
    if "model_type" not in old_config:
        old_config['model_type'] = 'tabpfn'
    if "num_predicted_hidden_layers" in old_config:
        old_config['predicted_hidden_layers'] = old_config.pop('num_predicted_hidden_layers')
    if "boolean_p_uninformative" in old_config:
        old_config['p_uninformative'] = old_config.pop('boolean_p_uninformative')
    if "boolean_max_fraction_uninformative" in old_config:
        old_config['max_fraction_uninformative'] = old_config.pop('boolean_max_fraction_uninformative')
    if old_config.pop("special_token", False):
        old_config['decoder_type'] = 'special_token'

    if old_config.pop("prenorm", False):
        print("prenorm is not supported anymore")
    if not old_config.pop("output_attention", True):
        raise NotImplementedError("output_attention=False is not supported anymore")
    if old_config.pop("decoder_two_hidden_layers", False):
        old_config['decoder_hidden_layers'] = 2
    else:
        old_config['decoder_hidden_layers'] = 1

    ignored_configs = ['seq_len_used', 'verbose', 'noise_type', 'normalize_to_ranking', 'normalize_by_used_features', 'num_categorical_features_sampler_a',
                       'differentiable', 'flexible', 'bptt_extra_samples', 'dynamic_batch_size', 'new_mlp_per_example', 'batch_size_per_gp_sample',
                       'normalize_ignore_label_too', 'differentiable_hps_as_style', 'rotate_normalized_labels', 'canonical_y_encoder',
                       'total_available_time_in_s', 'normalize_with_sqrt', 'done_part_in_training', 'mix_activations', 'save_every', 'create_new_run',
                       'perceiver_large_dataset', 'no_double_embedding', 'losses', 'wallclock_times', 'learning_rates', 'experiment', 'base_path',
                       'num_gpus', 'device', 'epoch_in_training', 'hid_factor', 'warm_start_from', 'continue_old_config', 'use_cpu', 'st_checkpoint_dir',
                       'use_mlflow', 'load_file', 'continue_run', 'load_strict', 'restart_scheduler', 'extra_fast_test', 'stop_after_epochs', 'shared_embedding',
                       'n_samples_used', 'double_embedding', 'learing_rate', 'gpu_id', 'agg_gradients', 'boolean_prior', 'seed_everything', 'model-type',
                       'num_features_used', 'max_eval_pos', 'nan_prob_unknown_reason_reason_prior', 'nan_prob_unknown_reason', 'no_mlflow']
    if old_config['model_type'] == 'tabpfn':
        # we used to store mothernet parameters in tabpfn models, but we no longer allow that
        ignored_configs.extend(['decoder_embed_dim', 'decoder_hidden_size', 'predicted_hidden_layer_size',
                                'predicted_hidden_layers', 'weight_embedding_rank', 'decoder_hidden_layers'])
    if old_config['model_type'] in ['mothernet', 'additive']:
        ignored_configs.extend(['num_latents'])
    for k in ignored_configs:
        old_config.pop(k, None)
    translated_config = nested_dict()
    for k, v in new_config.items():
        if k in old_config:
            translated_config[k] = old_config.pop(k)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        if k3 in old_config:
                            translated_config[k][k2][k3] = old_config.pop(k3)
                elif k2 in old_config:
                    translated_config[k][k2] = old_config.pop(k2)
    if len(old_config):
        raise ValueError(f"Unknown parameters: {old_config.keys()}")
    return translated_config


def _resolve_env_dim_upper_bound(spec, default=0):
    if spec is None:
        return int(default)
    if isinstance(spec, bool):
        return int(spec)
    if isinstance(spec, (int, float)):
        return int(spec)
    if isinstance(spec, dict):
        if "max" in spec:
            return int(spec["max"])
        if "value" in spec:
            return int(spec["value"])
        choice_values = spec.get("choice_values", None)
        if isinstance(choice_values, (list, tuple)) and len(choice_values) > 0:
            return int(max(choice_values))
    return int(default)


def get_model(
    config, 
    device, 
    should_train=True, 
    verbose=False, 
    model_state=None, 
    optimizer_state=None,
    scheduler=None, 
    epoch_callback=None, 
    load_model_strict=True
):
    passed_config = config.copy()

    # backwards compatibility for model names
    if 'model_type' not in passed_config:
        if 'model_maker' in passed_config:
            passed_config['model_type'] = passed_config.pop('model_maker')
        else:
            passed_config['model_type'] = 'tabpfn'

    if passed_config['model_type'] == 'mlp':
        passed_config['model_type'] = 'mothernet'
    config = get_model_default_config(passed_config['model_type'])

    if 'optimizer' not in passed_config:
        passed_config = old_config_to_new(passed_config, config)
    config.update(passed_config)
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    criterion = get_criterion(config['prior']['classification']['max_num_classes'])


    if 'transformer' in config:
        attention_type = 'transformer'
    elif 'linear_attention' in config:
        attention_type = 'linear_attention'
        if 'ssm' in passed_config:
            passed_config['linear_attention'] = passed_config.pop('ssm')
    else:
        raise ValueError(f"Unknown attention type")
    
    # backwards compatibility for cases where absence of parameter doesn't correspond to current default
    if 'n_samples' not in passed_config['prior']:
        config['prior']['n_samples'] = config['bptt']
    if 'y_encoder' not in passed_config[attention_type]:
        config[attention_type]['y_encoder'] = 'linear'

    if 'mothernet' in config:
        if 'decoder_activation' not in passed_config['mothernet']:
            config['mothernet']['decoder_activation'] = 'relu'
        if 'decoder_type' not in passed_config['mothernet']:
            config['mothernet']['decoder_type'] = 'output_attention'
        if (passed_config['mothernet'].get('weight_embedding_rank', None) is not None
                and 'low_rank_weights' not in passed_config['mothernet']):
            config['mothernet']['low_rank_weights'] = True

    y_encoder = get_y_encoder(config, attention_type)

    if config['prior']['classification']['max_num_classes'] > 2:
        n_out = config['prior']['classification']['max_num_classes']
    else:
        n_out = 1

    model_type = config['model_type']
    n_features = config['prior']['num_features']
    transformer_cfg = dict(config.get('transformer', {}))
    env_cfg = dict(config.get('prior', {}).get('environment', {}))
    backbone_kind = str(transformer_cfg.pop('backbone', 'transformer')).strip().lower()
    rwkv_head_size = transformer_cfg.pop('rwkv_head_size', 64)
    rwkv_ffn_mult = transformer_cfg.pop('rwkv_ffn_mult', 4)
    rwkv_sequence_replay_checkpoint = transformer_cfg.pop('rwkv_sequence_replay_checkpoint', False)
    rwkv_sequence_replay_batch_chunk_size = transformer_cfg.pop('rwkv_sequence_replay_batch_chunk_size', None)
    rwkv_sequence_replay_token_budget = transformer_cfg.pop('rwkv_sequence_replay_token_budget', None)
    normalized_q_value_head_enabled = bool(float(env_cfg.get('normalized_q_value_weight', 0.0) or 0.0) > 0.0)
    next_state_flow_dim = (
        _resolve_env_dim_upper_bound(env_cfg.get('state_dim', None), default=0)
        if float(env_cfg.get('next_state_flow_matching_weight', 0.0) or 0.0) > 0.0
        else None
    )
    default_next_state_flow_head_type = 'rwkv_two_layer' if backbone_kind == 'rwkv7' else 'cfmi_resnet'
    next_state_flow_head_type = str(
        env_cfg.get('next_state_flow_head_type', default_next_state_flow_head_type)
        or default_next_state_flow_head_type
    ).strip().lower()

    if model_type == "mothernet":
        model = MotherNet(
            n_out=n_out,
            y_encoder_layer=y_encoder, n_features=n_features, **transformer_cfg, **config['mothernet'])
    elif model_type == 'perceiver':
        model = TabPerceiver(n_out=n_out, y_encoder_layer=y_encoder, n_features=n_features,
                             **transformer_cfg, **config['mothernet'], **config['perceiver'])
    elif model_type == "additive":
        model = MotherNetAdditive(
            n_out=n_out, n_features=n_features,
            y_encoder_layer=y_encoder, **transformer_cfg, **config['mothernet'], **config['additive'])
    elif model_type in ["tabpfn", "rlpfn"]:
        if backbone_kind == "rwkv7":
            model = RWKV7PFN(
                n_out=n_out,
                n_features=n_features,
                y_encoder_layer=y_encoder,
                backbone=backbone_kind,
                rwkv_head_size=rwkv_head_size,
                rwkv_ffn_mult=rwkv_ffn_mult,
                rwkv_sequence_replay_checkpoint=rwkv_sequence_replay_checkpoint,
                rwkv_sequence_replay_batch_chunk_size=rwkv_sequence_replay_batch_chunk_size,
                rwkv_sequence_replay_token_budget=rwkv_sequence_replay_token_budget,
                normalized_q_value_head_enabled=normalized_q_value_head_enabled,
                next_state_flow_dim=next_state_flow_dim,
                next_state_flow_head_type=next_state_flow_head_type,
                **transformer_cfg,
            )
        else:
            model = TabPFN(
                n_out=n_out,
                n_features=n_features,
                y_encoder_layer=y_encoder,
                normalized_q_value_head_enabled=normalized_q_value_head_enabled,
                next_state_flow_dim=next_state_flow_dim,
                next_state_flow_head_type=next_state_flow_head_type,
                **transformer_cfg,
            )
    elif model_type == "batabpfn":
        # FIXME hack
        transformer_cfg['nhead'] = 4
        model = BiAttentionTabPFN(
            n_out=n_out, y_encoder_layer=y_encoder, **transformer_cfg, **config['biattention'])
    elif model_type == "baam":
        # FIXME hack
        transformer_cfg['nhead'] = 4
        model = GAMformer(
            n_out=n_out, n_features=config['prior']['num_features'],
            y_encoder_layer=y_encoder, **transformer_cfg, **config['mothernet'], **config['additive'])
    elif model_type in ['tabflex', 'ssm_tabpfn']:
        from ticl.models.tabflex import TabFlex
        config['linear_attention'].pop('causal_mask', None)
        model = TabFlex(
            n_out=n_out, 
            n_features=n_features, 
            y_encoder_layer=y_encoder, 
            **config['linear_attention']
        )
    elif model_type == 'la_mothernet':
        from ticl.models.la_mothernet import SSMMotherNet
        model = SSMMotherNet(
            n_out=n_out, 
            y_encoder_layer=y_encoder, 
            n_features=n_features, 
            **config['linear_attention'], 
            **config['mothernet']
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")

    if model_state is not None:
        prepare_state_dict_for_load = getattr(model, "prepare_state_dict_for_load", None)
        if callable(prepare_state_dict_for_load):
            model_state = prepare_state_dict_for_load(model_state, strict=load_model_strict)
        if not load_model_strict:
            for k, v in model.state_dict().items():
                if k in model_state and model_state[k].shape != v.shape:
                    model_state.pop(k)
        model.load_state_dict(model_state, strict=load_model_strict)

    if verbose:
        model_size = sum(p.numel() for p in model.parameters())
        if wandb.run: wandb.log({"model_size": model_size})
        if 'linear_attention' in model_type:
            print(f"Using a SSM with {model_size/1000/1000:.{2}f} M parameters")
        else:
            print(f"Using a Transformer with {model_size/1000/1000:.{2}f} M parameters")

    if 'losses' in config:
        # for continuing training
        model.losses = config['losses']
        model.learning_rates = config['learning_rates']
        model.wallclock_times = config.get('wallclock_times', [])

    if should_train:
        model_attr = None if 'linear_attention' not in model_type else config['linear_attention']['model']
        dl = get_dataloader(
            prior_config=config['prior'], 
            dataloader_config=config['dataloader'], 
            device=device,
            model = model_attr,
        )
        model = train(dl, model, criterion=criterion, optimizer_state=optimizer_state, scheduler=scheduler,
                      epoch_callback=epoch_callback, verbose=verbose_train, device=device, progress_bar=config['orchestration']['progress_bar'], **config['optimizer'])
    else:
        model = None, model, None, None

    return model
