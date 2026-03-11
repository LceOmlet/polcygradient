import argparse
from ticl.config_utils import str2bool
from ticl.model_configs import get_model_default_config
from ticl.rl_validation import RLPFN_DEFAULT_OOP_ENVS


class GroupedArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
    # This extends the argparse.ArgumentParser to allow for nested namespaces via groups
    # nesting of groups is done by giving them names with dots in them

    def parse_known_args(self, args=None, namespace=None):
        results, args = super().parse_known_args(args=args, namespace=namespace)
        nested_by_groups = argparse.Namespace()
        for group in self._action_groups:
            # group could have been created if we saw a nested group first
            new_subnamespace = getattr(nested_by_groups, group.title, argparse.Namespace())
            for action in group._group_actions:
                if action.dest is not argparse.SUPPRESS and hasattr(results, action.dest):
                    setattr(new_subnamespace, action.dest, getattr(results, action.dest))
            if new_subnamespace != argparse.Namespace():
                parts = group.title.split(".")
                parent_namespace = nested_by_groups
                for part in parts[:-1]:
                    if not hasattr(parent_namespace, part):
                        setattr(parent_namespace, part, argparse.Namespace())
                    parent_namespace = getattr(parent_namespace, part)
                setattr(parent_namespace, parts[-1], new_subnamespace)

        return nested_by_groups, args


def make_model_level_argparser(description="Train transformer-style model on synthetic data"):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(required=True, parser_class=GroupedArgParser,
                                       description="Choose the model type to train.", dest='model_type')
    mothernet_parser = subparsers.add_parser('mothernet', help='Train a mothernet model')
    mothernet_parser.set_defaults(model_type='mothernet')
    mothernet_parser = argparser_from_config(description="Train Mothernet", parser=mothernet_parser)

    tabpfn_parser = subparsers.add_parser('tabpfn', help='Train a tabpfn model')
    tabpfn_parser.set_defaults(model_type='tabpfn')
    tabpfn_parser = argparser_from_config(description="Train tabpfn", parser=tabpfn_parser)

    rlpfn_parser = subparsers.add_parser('rlpfn', help='Train an rlpfn model')
    rlpfn_parser.set_defaults(model_type='rlpfn')
    rlpfn_parser = argparser_from_config(description="Train rlpfn", parser=rlpfn_parser)

    additive_parser = subparsers.add_parser('additive', help='Train an additive mothernet model')
    additive_parser.set_defaults(model_type='additive')
    additive_parser = argparser_from_config(description="Train additive", parser=additive_parser)

    perceiver_parser = subparsers.add_parser('perceiver', help='Train a perceiver mothernet model')
    perceiver_parser.set_defaults(model_type='perceiver')
    perceiver_parser = argparser_from_config(description="Train perceiver", parser=perceiver_parser)

    batabpfn_parser = subparsers.add_parser('batabpfn', help='Train a bi-attention tabpfn model')
    batabpfn_parser.set_defaults(model_type='batabpfn')
    batabpfn_parser = argparser_from_config(description="Train batabpfn", parser=batabpfn_parser)

    baam_parser = subparsers.add_parser('baam', help='Train a bi-attention additive mothernet model')
    baam_parser.set_defaults(model_type='baam')
    baam_parser = argparser_from_config(description="Train baam", parser=baam_parser)
    
    tabflex_parser = subparsers.add_parser('tabflex', help='Train a TabFlex model')
    tabflex_parser.set_defaults(model_type='tabflex')
    tabflex_parser = argparser_from_config(description="Train TabFlex", parser=tabflex_parser)

    la_mothernet_parser = subparsers.add_parser('la_mothernet', help='Train a la_mothernet model')
    la_mothernet_parser.set_defaults(model_type='la_mothernet')
    la_mothernet_parser = argparser_from_config(description="Train SSMMothernet", parser=la_mothernet_parser)

    return parser


def argparser_from_config(parser, description="Train Mothernet"):
    model_type = parser.get_default('model_type')
    config = get_model_default_config(model_type)
    # all models have general, optimizer, dataloader and transformer parameters
    general = parser.add_argument_group('general')
    general.add_argument('-g', '--gpu-id', type=int, help='GPU id')
    general.add_argument('-C', '--use-cpu', help='whether to use cpu', action='store_true')

    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('-E', '--epochs', type=int, help='number of epochs')
    optimizer.add_argument('-l', '--learning-rate', type=float, help='maximum learning rate')
    optimizer.add_argument('-k', '--aggregate_k_gradients', type=int, help='number steps to aggregate gradient over')
    optimizer.add_argument('--rl-objective', type=str, choices=['supervised', 'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad'],
                           help='Training objective for RL-style models.')
    optimizer.add_argument('--policy-rollout-chunk-size', type=int,
                           help='Policy-gradient rollout chunk size over batch columns. None uses auto(batch_size); <=0 forces full batch.')
    optimizer.add_argument('--policy-rollout-chunk-autotune', type=str2bool,
                           help='Auto-tune policy rollout chunk size after OOM by gradually growing it back on stable batches.')
    optimizer.add_argument('--policy-rollout-chunk-grow-every', type=int,
                           help='When chunk auto-tune is enabled, number of stable batches before growing chunk size.')
    optimizer.add_argument('--policy-rollout-chunk-grow-factor', type=float,
                           help='When chunk auto-tune is enabled, multiplicative factor for chunk-size growth.')
    optimizer.add_argument('--policy-rollout-checkpoint', type=str2bool,
                           help='Enable activation checkpointing for policy-gradient rollout chunks (recompute forward in backward to reduce peak memory).')
    optimizer.add_argument('--policy-rollout-checkpoint-reentrant', type=str2bool,
                           help='Use reentrant checkpoint for rollout chunks (typically lower memory, higher recompute overhead).')
    optimizer.add_argument('--pg-grad-mutable-kv-cache', type=str2bool,
                           help='Allow grad-enabled mutable KV cache during policy-gradient rollout (typically with paged mode). Only active with rollout checkpoint.')
    optimizer.add_argument('--pg-saved-tensors-cpu-offload', type=str2bool,
                           help='Offload autograd saved tensors to CPU during policy-gradient rollout (lower GPU memory, slower).')
    optimizer.add_argument('--pg-saved-tensors-pin-memory', type=str2bool,
                           help='When CPU-offloading saved tensors, use pinned host memory for faster H2D transfers.')
    optimizer.add_argument('--pg-oom-debug-raise', type=str2bool,
                           help='When true, re-raise policy-gradient OOM exceptions immediately for full traceback debugging.')
    optimizer.add_argument('--pg-oom-fail-fast', type=str2bool,
                           help='When true, disable all policy-gradient OOM fallback (TBPTT/chunk) and fail immediately.')
    optimizer.add_argument('--pg-kv-cache-mode', type=str, choices=['auto', 'immutable', 'static', 'paged'],
                           help='KV-cache mode used by policy forward_step during policy-gradient rollout.')
    optimizer.add_argument('--pg-kv-cache-page-size', type=int,
                           help='Paged KV-cache page size for policy-gradient rollout when pg-kv-cache-mode=paged.')
    optimizer.add_argument('--pg-tbptt-window', type=int,
                           help='Truncated-BPTT window length over rollout time steps; None/<=0 keeps full-horizon policy-gradient semantics.')
    optimizer.add_argument('--pg-env-replay-steps', type=int,
                           help='For policy-gradient objective, number of sequential rollout->optimize updates to run on the same sampled batch environment before sampling the next batch environment.')
    optimizer.add_argument('--pg-oom-reduce-tbptt-first', type=str2bool,
                           help='When policy-gradient OOM fallback is enabled, reduce TBPTT window before shrinking rollout chunk size.')
    optimizer.add_argument('--pg-torch-compile', type=str2bool,
                           help='Compile policy step forward with torch.compile for policy-gradient training.')
    optimizer.add_argument('--pg-torch-compile-backend', type=str,
                           help='torch.compile backend for policy step (e.g., inductor, eager).')
    optimizer.add_argument('--pg-torch-compile-mode', type=str,
                           help='torch.compile mode for policy step (e.g., reduce-overhead, max-autotune).')
    optimizer.add_argument('--pg-torch-compile-fullgraph', type=str2bool,
                           help='Use fullgraph mode when compiling policy step.')
    optimizer.add_argument('--pg-torch-compile-dynamic', type=str2bool,
                           help='Enable dynamic-shape compile for policy step.')
    optimizer.add_argument('--pg-compile-observe-recompiles', type=str2bool,
                           help='Enable per-batch torch.compile/torch._dynamo counter delta logging (recompile observability).')
    optimizer.add_argument('--pg-compile-observe-log-every-batches', type=int,
                           help='When compile observability is enabled, emit counter deltas every N batches.')
    optimizer.add_argument('--pg-compile-observe-output-path', type=str,
                           help='Optional JSONL path for per-batch compile counter deltas.')
    optimizer.add_argument('--pg-compile-observe-reset-after-warmup', type=str2bool,
                           help='When compile warmup runs, reset compile-counter baseline after warmup to remove warmup pollution.')
    optimizer.add_argument('--adamw-fused', type=str2bool,
                           help='Use fused AdamW on CUDA when supported.')
    optimizer.add_argument('--train-profiler-enabled', type=str2bool,
                           help='Enable structured per-epoch training profiler.')
    optimizer.add_argument('--train-profiler-output-path', type=str,
                           help='Optional JSONL output path for training profiler records.')
    optimizer.add_argument('--train-profiler-wandb', type=str2bool,
                           help='When profiler is enabled, log profiler records to wandb with profile/* keys.')
    optimizer.add_argument('--train-profiler-ema-alpha', type=float,
                           help='EMA alpha for smoothed profiler throughput metrics.')
    optimizer.add_argument('--train-profiler-warmup-epochs', type=int,
                           help='Number of initial epochs excluded from EMA smoothing.')
    optimizer.add_argument('--train-profiler-warmup-batches', type=int,
                           help='Number of initial batches excluded from EMA smoothing.')
    optimizer.add_argument('--train-profiler-log-every-batches', type=int,
                           help='Emit interval profiler records every N batches (0 disables interval records).')
    optimizer.add_argument('--train-gpu-observer-enabled', type=str2bool,
                           help='Enable background GPU observer sampling for per-batch/stage JSONL observability.')
    optimizer.add_argument('--train-gpu-observer-interval-sec', type=float,
                           help='Sampling period in seconds for GPU observer.')
    optimizer.add_argument('--train-gpu-observer-output-path', type=str,
                           help='Optional JSONL output path for raw GPU observer samples.')
    optimizer.add_argument('--train-gpu-stage-output-path', type=str,
                           help='Optional JSONL output path for stage-window GPU observer records.')
    optimizer.add_argument('--train-kernel-profiler-enabled', type=str2bool,
                           help='Enable torch.profiler kernel-level tracing during training.')
    optimizer.add_argument('--train-kernel-profiler-output-dir', type=str,
                           help='Directory for torch.profiler traces and kernel summaries.')
    optimizer.add_argument('--train-kernel-profiler-wait-steps', type=int,
                           help='Profiler schedule wait steps.')
    optimizer.add_argument('--train-kernel-profiler-warmup-steps', type=int,
                           help='Profiler schedule warmup steps.')
    optimizer.add_argument('--train-kernel-profiler-active-steps', type=int,
                           help='Profiler schedule active steps.')
    optimizer.add_argument('--train-kernel-profiler-repeat-steps', type=int,
                           help='Profiler schedule repeat count.')
    optimizer.add_argument('--train-kernel-profiler-record-shapes', type=str2bool,
                           help='Record operator input shapes in kernel profiler.')
    optimizer.add_argument('--train-kernel-profiler-profile-memory', type=str2bool,
                           help='Record memory profiling info in kernel profiler.')
    optimizer.add_argument('--train-kernel-profiler-with-stack', type=str2bool,
                           help='Capture Python stack for profiler events (higher overhead).')
    optimizer.add_argument('--train-kernel-profiler-with-flops', type=str2bool,
                           help='Collect FLOPs estimates in profiler when available.')
    optimizer.add_argument('--train-kernel-profiler-log-every-batches', type=int,
                           help='Emit top-op kernel summary every N batches (0 disables summary emission).')
    optimizer.add_argument('--train-kernel-profiler-export-trace', type=str2bool,
                           help='Export tensorboard trace files for kernel profiler. Disable for low-overhead summary-only profiling.')
    optimizer.add_argument('--train-kernel-profiler-summary-top-k', type=int,
                           help='Number of top operators to keep in each kernel-profiler summary record.')
    optimizer.add_argument('-A', '--adaptive-batch-size', help='Wether to progressively increase effective batch size.',
                           type=str2bool)
    optimizer.add_argument('-w', '--weight-decay', type=float, help='Weight decay for AdamW.')
    optimizer.add_argument('-Q', '--learning-rate-schedule', help="Learning rate schedule. Cosine, constant or exponential")
    optimizer.add_argument('-U', '--warmup-epochs', type=int, help="Number of epochs to warm up learning rate (linear climb)")
    optimizer.add_argument('-t', '--train-mixed-precision', help='whether to train with mixed precision', type=str2bool)
    optimizer.add_argument('--adam-beta1', type=float)
    optimizer.add_argument('--lr-decay', help="learning rate decay when using exponential schedule", type=float)
    optimizer.add_argument('--min-lr', help="minimum learning rate for any schedule", type=float)
    optimizer.add_argument('--reduce-lr-on-spike', help="Whether to half learning rate when observing a loss spike", type=str2bool)
    optimizer.add_argument('--spike-tolerance', help="how many times the std makes it a spike", type=int)
    optimizer.set_defaults(**config['optimizer'])

    dataloader = parser.add_argument_group('dataloader')
    dataloader.add_argument('-b', '--batch-size', type=int, help='physical batch size')
    dataloader.add_argument('-n', '--num-steps', type=int, help='number of steps per epoch')
    dataloader.add_argument('--min-eval-pos', type=int, help='minimum evaluation position')
    dataloader.add_argument('--random-n-samples', type=int, help='whether to sample n_samples randomly')
    dataloader.add_argument('--n-test-samples', type=int, help='number of test samples')
    dataloader.set_defaults(**config['dataloader'])
    
    openmlloader = parser.add_argument_group('openmlloader')
    openmlloader.add_argument('--valid-data', help='whether to use large dataset', choices=['new', 'large', 'old'])
    openmlloader.add_argument('--pca', help='whether to use pca', action='store_true')
    openmlloader.set_defaults(**config['openmlloader'])


    if 'transformer' in config:
        transformer = parser.add_argument_group('transformer')
        transformer.add_argument('-e', '--emsize', type=int, help='embedding size')
        transformer.add_argument('-N', '--nlayers', type=int, help='number of transformer layers')
        transformer.add_argument('--init-method', help='Weight initialization method.')
        transformer.add_argument('--y-encoder', help='Encoder for labels. "linear", "onehot" or None.')
        transformer.add_argument('--tabpfn-zero-weights', help='Whether to use zeroing of weights from tabpfn code.', type=str2bool)
        transformer.add_argument('--pre-norm', action='store_true')
        transformer.add_argument('--classification-task', type=str2bool, help='Whether to use classification or regression.')
        transformer.add_argument('--x-encoder-type', choices=['single', 'split_obs_action'],
                                 help='X encoder layout: single head or split obs/action heads.')
        transformer.add_argument('--x-obs-dim', type=int, help='Input width for obs/reward/mask head when using split encoder.')
        transformer.add_argument('--x-action-dim', type=int, help='Input width for action head when using split encoder.')
        transformer.add_argument('--single-eval-causal', type=str2bool,
                                 help='Enable causal single-eval path with KV-cache inference.')
        transformer.set_defaults(**config['transformer'])
    elif 'linear_attention' in config:
        linear_attention = parser.add_argument_group('linear_attention')
        linear_attention.add_argument('-e', '--emsize', type=int, help='embedding size')
        linear_attention.add_argument('-N', '--nlayers', type=int, help='number of transformer layers')
        linear_attention.add_argument('--init-method', help='Weight initialization method.')
        linear_attention.add_argument('--y-encoder', help='Encoder for labels. "linear", "onehot" or None.')
        linear_attention.add_argument('--tabpfn-zero-weights', help='Whether to use zeroing of weights from tabpfn code.', type=str2bool)
        linear_attention.add_argument('--pre-norm', action='store_true')
        linear_attention.add_argument('--classification-task', type=str2bool, help='Whether to use classification or regression.')
        linear_attention.add_argument('--model', type = str, choices = ['linear_attention', 'fla'], help = 'which linear_attention model to use')
        
        ## specific to fla
        linear_attention.add_argument('--feature-map', help='when the model is fla, which feature map to use', type = str, choices = ['identity_for_real', 'elu', 'hedgehog', 'hedgehog_shared'])
        linear_attention.add_argument('--norm-output', help='when the model is fla, whether to normalize the output of the model', action = 'store_true', default = False)
        linear_attention.add_argument('--causal-mask', help='when the model is fla, Whether to use causal attention', action='store_true', default=False)
        linear_attention.set_defaults(**config['linear_attention'])
    else:
        raise ValueError("No transformer or linear_attention config found in model config.")

    if model_type in ['baam', 'batabpfn']:
        biattention = parser.add_argument_group('biattention')
        biattention.add_argument('--input-embedding', type=str, help='input embedding type')
        biattention.set_defaults(**config['biattention'])

    if model_type in ['mothernet', 'additive', 'baam', 'perceiver', 'la_mothernet']:
        mothernet = parser.add_argument_group('mothernet')
        mothernet.add_argument('-d', '--decoder-embed-dim', type=int, help='decoder embedding size')
        mothernet.add_argument('-H', '--decoder-hidden-size', type=int, help='decoder hidden size')
        mothernet.add_argument('--decoder-activation', type=str, help='decoder activation')
        mothernet.add_argument('-D', '--decoder-type',
                               help="Decoder Type. 'output_attention', 'special_token', 'class_average' or 'average'.", type=str)
        mothernet.add_argument('-T', '--decoder-hidden-layers', help='How many hidden layers to use in decoder MLP', type=int)
        mothernet.add_argument('-P', '--predicted-hidden-layer-size', type=int, help='Size of hidden layers in predicted network.')
        mothernet.add_argument('-L', '--predicted-hidden-layers', type=int, help='number of predicted hidden layers')
        mothernet.add_argument('--predicted-activation', type=str, help="activation in predicted network")
        mothernet.add_argument('-r', '--low-rank-weights', type=str2bool, help='Whether to use low-rank weights in mothernet.')
        mothernet.add_argument('-W', '--weight-embedding-rank', type=int, help='Rank of weights in predicted network.')
        mothernet.set_defaults(**config['mothernet'])

    if model_type in ['additive', 'baam']:
        additive = parser.add_argument_group('additive')
        additive.add_argument('--input-bin-embedding',
                              help="'linear' for linear bin embedding, 'non-linear' for nonlinear, 'none' or False for no embedding.", type=str)
        additive.add_argument('--bin-embedding-rank', help="Rank of bin embedding", type=int)
        additive.add_argument('--fourier-features', help="Number of Fourier features to add per feature. A value of 0 means off.", type=int)
        additive.add_argument('--n-bins', help="Number of bins", type=int)
        additive.add_argument('--nan-bin', help="Whether to use the last bin to denote a nan value.", type=str2bool)
        additive.add_argument('--sklearn-binning', help="Whether to bin the features with less num bins features using sklearn method.", type=str2bool)
        additive.add_argument('--categorical-embedding', help="Whether to embed the categorical features using a separate embedding", type=str2bool)
        additive.add_argument('--marginal-residual', help="Whether to learn the residual of the marginals. 'output', 'decoder' or 'none'.", type=str)
        additive.add_argument('--factorized-output', help="whether to use a factorized output", type=str2bool)
        additive.add_argument('--output-rank', help="Rank of output in factorized output", type=int)
        additive.add_argument('--input-layer-norm', help="Whether to use layer norm on one-hot encoded data.", type=str2bool)
        additive.add_argument('--shape-attention', help="Whether to use attention in low rank output.", type=str2bool)
        additive.add_argument('--shape-attention-heads', help="Number of heads in shape attention.", type=int)
        additive.add_argument('--n-shape-functions', help="Number of shape functions in shape attention.", type=int)
        additive.add_argument('--shape-init', help="How to initialize shape functions. 'constant' for unit variance, 'inverse' for 1/(n_shape_functions * n_bins), "
                              "'sqrt' for 1/sqrt(n_shape_functions * n_bins). 'inverse_bins' for 1/n_bins, 'inverse_sqrt_bins' for 1/sqrt(n_bins)",
                              type=str)
        additive.set_defaults(**config['additive'])

    if model_type in ['perceiver']:
        perceiver = parser.add_argument_group('perceiver')
        perceiver.add_argument('--num-latents', help="number of latent variables in perceiver", type=int)
        # perceiver.add_argument('--perceiver-large-dataset', action='store_true')
        perceiver.set_defaults(**config['perceiver'])


    # Prior and data generation
    prior = parser.add_argument_group('prior')
    prior.add_argument('--num-features', help="Maximum number of features in prior", type=int)
    prior.add_argument('--n-samples', help="Maximum Number of samples in prior", type=int)
    prior.add_argument('--prior-type', help="Which prior to use, available ['prior_bag', 'environment_only', 'boolean_only', 'bag_boolean', 'step_function'].", type=str)
    prior.set_defaults(**config['prior'])

    classification_prior = parser.add_argument_group('prior.classification')
    classification_prior.add_argument('--multiclass-type', help="Which multiclass prior to use ['steps', 'rank'].", type=str)
    classification_prior.add_argument('--num-features-sampler', help="How to sample number of features, 'fixed', 'uniform', or 'double_sample'. ", type=str)
    classification_prior.add_argument('--multiclass-max-steps', help="Maximum number of steps in multiclass step prior", type=int)
    classification_prior.add_argument('--pad-zeros', help="Whether to pad data with zeros for consistent size", type=str2bool)
    classification_prior.add_argument('--max-num-classes', help="Maximum number of classes. 0 means regression.", type=int)
    classification_prior.add_argument('--nan-prob-no-reason', help="NaN probability missing at random.", type=float)
    classification_prior.add_argument('--nan-prob-a-reason', help="NaN probability missing not at random.", type=float)
    classification_prior.add_argument('--categorical-feature-p', help="Categorical feature probability.", type=float)
    classification_prior.add_argument('--feature-curriculum', help="Whether to use a curriculum for number of features", type=str2bool)
    classification_prior.set_defaults(**config['prior']['classification'])

    mlp_prior = parser.add_argument_group('prior.mlp')
    mlp_prior.add_argument('--add-uninformative-features', help="Whether to add uniformative features in the MLP prior.", type=str2bool)
    mlp_prior.set_defaults(**config['prior']['mlp'])

    environment_prior = parser.add_argument_group('prior.environment')
    environment_prior.add_argument('--family', type=str, choices=['scm', 'gp'],
                                   help='Environment generator family: scm or gp.')
    environment_prior.add_argument('--action-dim', type=int, help='Fixed action dimension when overriding sampled range.')
    environment_prior.add_argument('--state-dim', type=int, help='Fixed latent state dimension when overriding sampled range.')
    environment_prior.add_argument('--obs-dim', type=int, help='Fixed observed state dimension when overriding sampled range.')
    environment_prior.add_argument('--noise-dim', type=int, help='Fixed transition-noise dimension when overriding sampled range.')
    environment_prior.add_argument('--zero-pad-dim', type=int, help='Fixed zero-pad dimension in env input.')
    environment_prior.add_argument('--obs-slot-dim', type=int, help='Fixed observation slot width before reward/mask append.')
    environment_prior.add_argument('--action-slot-dim', type=int, help='Fixed action slot width in PFN input token.')
    environment_prior.add_argument('--alpha', type=float, help='State update mixing coefficient.')
    environment_prior.add_argument('--init-state-std', type=float, help='Std for Gaussian initialization of s0.')
    environment_prior.add_argument('--init-action-std', type=float, help='Std for Gaussian initialization of a0.')
    environment_prior.add_argument('--state-noise-std', type=float, help='Std for additive state noise per step.')
    environment_prior.add_argument('--action-noise-train-std', type=float, help='Std for action noise before eval split.')
    environment_prior.add_argument('--action-noise-eval-std', type=float, help='Std for action noise after eval split.')
    environment_prior.add_argument('--reward-scale', type=float, help='Scale multiplier for sampled rewards.')
    environment_prior.add_argument('--reward-clip', type=float, help='Absolute clip bound applied to sampled rewards.')
    environment_prior.add_argument('--state-clip', type=float, help='Clamp bound for latent state before tanh.')
    environment_prior.add_argument('--policy-gradient-normalize-rewards', type=str2bool,
                                   help='If true, optimize normalized rewards; if false, optimize raw discounted reward mean.')
    environment_prior.add_argument('--reward-norm-eps', type=float, help='Epsilon for reward normalization.')
    environment_prior.add_argument('--reward-norm-clip', type=float, help='Clip bound for normalized rewards.')
    environment_prior.add_argument('--discount', type=float, help='Discount factor for policy-gradient objective.')
    environment_prior.add_argument('--first-policy-gradient-state-grad-clip-norm', type=float,
                                   help='Per-sample global-norm clip applied to environment state adjoints for first_policy_gradient/alpha_grad.')
    environment_prior.add_argument('--first-policy-gradient-action-grad-clip-value', type=float,
                                   help='Elementwise absolute clip applied to environment action adjoints for first_policy_gradient/alpha_grad.')
    environment_prior.add_argument('--first-policy-gradient-action-grad-clip-norm', type=float,
                                   help='Per-sample global-norm clip applied to environment action adjoints for first_policy_gradient/alpha_grad.')
    environment_prior.add_argument('--anti-explosion-vanishing-v2-enabled', type=str2bool,
                                   help='Enable anti-explosion&vanishing-v2 (two-sided state-gain corridor regularization).')
    environment_prior.add_argument('--anti-explosion-vanishing-v2-lambda', type=float,
                                   help='Regularization weight for anti-explosion&vanishing-v2.')
    environment_prior.add_argument('--anti-explosion-vanishing-v2-gain-lo', type=float,
                                   help='Lower gain corridor bound for anti-explosion&vanishing-v2.')
    environment_prior.add_argument('--anti-explosion-vanishing-v2-gain-hi', type=float,
                                   help='Upper gain corridor bound for anti-explosion&vanishing-v2.')
    environment_prior.add_argument('--anti-explosion-vanishing-v2-huber-delta', type=float,
                                   help='Huber delta for corridor-violation penalty in anti-explosion&vanishing-v2.')
    environment_prior.add_argument('--anti-explosion-vanishing-v2-eps', type=float,
                                   help='Numerical epsilon for anti-explosion&vanishing-v2.')
    environment_prior.add_argument('--anti-explosion-vanishing-v2-detach-reference', type=str2bool,
                                   help='Detach previous-step increment norm in anti-explosion&vanishing-v2 gain computation.')
    environment_prior.add_argument('--anti-explosion-vanishing-v3-enabled', type=str2bool,
                                   help='Enable anti-explosion&vanishing-v3 (drift+tail log-gain regularization).')
    environment_prior.add_argument('--anti-explosion-vanishing-v3-lambda-drift', type=float,
                                   help='Drift regularization weight for anti-explosion&vanishing-v3.')
    environment_prior.add_argument('--anti-explosion-vanishing-v3-lambda-tail', type=float,
                                   help='Tail regularization weight for anti-explosion&vanishing-v3.')
    environment_prior.add_argument('--anti-explosion-vanishing-v3-gain-lo', type=float,
                                   help='Lower gain corridor bound for anti-explosion&vanishing-v3.')
    environment_prior.add_argument('--anti-explosion-vanishing-v3-gain-hi', type=float,
                                   help='Upper gain corridor bound for anti-explosion&vanishing-v3.')
    environment_prior.add_argument('--anti-explosion-vanishing-v3-tail-tau', type=float,
                                   help='Softplus temperature for anti-explosion&vanishing-v3 tail penalty.')
    environment_prior.add_argument('--anti-explosion-vanishing-v3-eps', type=float,
                                   help='Numerical epsilon for anti-explosion&vanishing-v3.')
    environment_prior.add_argument('--anti-explosion-vanishing-v3-detach-reference', type=str2bool,
                                   help='Detach previous-step increment norm in anti-explosion&vanishing-v3 gain computation.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-enabled', type=str2bool,
                                   help='Enable anti-explosion&vanishing-v4 (controlled highway update + gain regularization).')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-lambda-drift', type=float,
                                   help='Drift regularization weight for anti-explosion&vanishing-v4.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-lambda-tail', type=float,
                                   help='Tail regularization weight for anti-explosion&vanishing-v4.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-gain-lo', type=float,
                                   help='Lower gain corridor bound for anti-explosion&vanishing-v4.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-gain-hi', type=float,
                                   help='Upper gain corridor bound for anti-explosion&vanishing-v4.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-tail-tau', type=float,
                                   help='Softplus temperature for anti-explosion&vanishing-v4 tail penalty.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-eps', type=float,
                                   help='Numerical epsilon for anti-explosion&vanishing-v4.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-detach-reference', type=str2bool,
                                   help='Detach previous-step update norm in anti-explosion&vanishing-v4 gain computation.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-highway-ratio', type=float,
                                   help='Fraction of state dims used by anti-explosion&vanishing-v4 highway update.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-update-scale', type=float,
                                   help='Residual update scale on the v4 highway subspace.')
    environment_prior.add_argument('--anti-explosion-vanishing-v4-update-clip', type=float,
                                   help='Absolute clip bound for v4 highway residual update (<=0 disables clipping).')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-enabled', type=str2bool,
                                   help='Enable anti-explosion&vanishing-v5 (detached reward-signal thermostat).')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-target-std', type=float,
                                   help='Target reward std used by anti-explosion&vanishing-v5 loss scaling.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-scale-lo', type=float,
                                   help='Lower bound of detached loss scale in anti-explosion&vanishing-v5.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-scale-hi', type=float,
                                   help='Upper bound of detached loss scale in anti-explosion&vanishing-v5.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-eps', type=float,
                                   help='Numerical epsilon for anti-explosion&vanishing-v5 std reference.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-detach-reference', type=str2bool,
                                   help='Detach reward std reference in anti-explosion&vanishing-v5 scaling.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-enabled', type=str2bool,
                                   help='Enable anti-explosion&vanishing-v5_next (full-state corridor + detached thermostat).')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-state-gain-lo', type=float,
                                   help='Lower directional gain corridor for anti-explosion&vanishing-v5_next state updates.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-state-gain-hi', type=float,
                                   help='Upper directional gain corridor for anti-explosion&vanishing-v5_next state updates.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-state-rms-lo', type=float,
                                   help='Lower RMS corridor for anti-explosion&vanishing-v5_next state updates.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-state-rms-hi', type=float,
                                   help='Upper RMS corridor for anti-explosion&vanishing-v5_next state updates.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-state-reward-gate', type=float,
                                   help='Reward-magnitude gate for low-side anti-explosion&vanishing-v5_next state protection.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-state-low-boost-cap', type=float,
                                   help='Maximum low-side boost for anti-explosion&vanishing-v5_next state updates.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-loss-target-std', type=float,
                                   help='Target reward std used by anti-explosion&vanishing-v5_next loss scaling.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-loss-scale-lo', type=float,
                                   help='Lower bound of detached loss scale in anti-explosion&vanishing-v5_next.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-loss-scale-hi', type=float,
                                   help='Upper bound of detached loss scale in anti-explosion&vanishing-v5_next.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-step-grad-rms-lo', type=float,
                                   help='Lower train-step gradient RMS corridor for anti-explosion&vanishing-v5_next.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-step-grad-rms-hi', type=float,
                                   help='Upper train-step gradient RMS corridor for anti-explosion&vanishing-v5_next.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-step-reward-std-gate', type=float,
                                   help='Reward-std gate for low-side anti-explosion&vanishing-v5_next train-step protection.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-step-low-boost-cap', type=float,
                                   help='Maximum low-side boost for anti-explosion&vanishing-v5_next train-step scaling.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-eps', type=float,
                                   help='Numerical epsilon for anti-explosion&vanishing-v5_next.')
    environment_prior.add_argument('--anti-explosion-vanishing-v5-next-detach-reference', type=str2bool,
                                   help='Detach corridor references in anti-explosion&vanishing-v5_next.')
    environment_prior.add_argument('--lipschitz-enforce', type=str2bool,
                                   help='Enable Lipschitz safeguards for sampled environment generators.')
    environment_prior.add_argument('--lipschitz-weight-fro-norm-max', type=float,
                                   help='Per-layer Frobenius-norm cap for sampled SCM/GP linear maps.')
    environment_prior.add_argument('--lipschitz-gp-outputscale-max', type=float,
                                   help='Absolute cap for GP outputscale used by sampled transition/reward generators.')
    environment_prior.add_argument('--num-layers', type=int, help='Depth for scm generator network.')
    environment_prior.add_argument('--prior-mlp-hidden-dim', type=int, help='Hidden dim for scm generator network.')
    environment_prior.add_argument('--prior-mlp-activations', type=str, choices=['tanh', 'relu', 'identity'],
                                   help='Activation for scm generator network.')
    environment_prior.add_argument('--init-std', type=float, help='Weight init std for scm generator network.')
    environment_prior.add_argument('--noise-std', type=float, help='Output noise std for scm generator network.')
    environment_prior.add_argument('--lengthscale', type=float, help='GP lengthscale for gp family.')
    environment_prior.add_argument('--outputscale', type=float, help='GP output scale for gp family.')
    environment_prior.add_argument('--noise', type=float, help='GP observation noise for gp family.')
    environment_prior.add_argument('--gp-rff-features', type=int, help='Number of random Fourier features for gp family.')
    environment_prior.add_argument('--reward-dropout-enabled', type=str2bool, help='Enable reward dropout masking in prior tokens.')
    environment_prior.add_argument('--reward-dropout-randomize', type=str2bool, help='Sample reward dropout ratio per environment.')
    environment_prior.add_argument('--reward-dropout-ratio', type=float, help='Fixed reward dropout ratio when randomization is off.')
    environment_prior.add_argument('--reward-dropout-ratio-min', type=float, help='Minimum reward dropout ratio when randomizing.')
    environment_prior.add_argument('--reward-dropout-ratio-max', type=float, help='Maximum reward dropout ratio when randomizing.')
    environment_prior.add_argument('--reward-dropout-impute-zero', type=str2bool, help='Use zero imputation for dropped rewards.')
    environment_prior.add_argument('--batch-parallel-workers', type=int, help='Parallel workers for independent per-column rollout in get_batch.')
    environment_prior.add_argument('--batch-parallel-backend', type=str, choices=['python_thread', 'torch_vectorized'],
                                   help='Backend for batch generation parallelism in environment prior.')
    environment_prior.add_argument('--batch-shared-environment', type=str2bool,
                                   help='Whether all columns in a batch share one sampled environment function.')
    environment_prior.add_argument('--batch-vectorized-strict-rng-match', type=str2bool,
                                   help='Match serial RNG stream exactly in torch_vectorized backend (slower; for A/B tests).')
    environment_prior.add_argument('--batch-vectorized-grouping', type=str, choices=['structure', 'family'],
                                   help='Grouping strategy for torch_vectorized backend.')
    environment_prior.set_defaults(**config['prior']['environment'])

    boolean = parser.add_argument_group('prior.boolean')
    boolean.add_argument('--p-uninformative', help="Probability of adding uninformative features in boolean prior",
                         type=float)
    boolean.add_argument('--max-fraction-uninformative', help="Maximum fraction opf uninformative features in boolean prior",
                         type=float)
    boolean.set_defaults(**config['prior']['boolean'])

    # serialization, loading, logging
    orchestration = parser.add_argument_group('orchestration')
    orchestration.add_argument('--extra-fast-test', help="whether to use tiny data", action='store_true')
    orchestration.add_argument('--stop-after-epochs', help="for pausing rungs with synetune", type=int, default=None)
    orchestration.add_argument('--seed-everything', help="whether to seed everything for testing and benchmarking", default = False, type=str2bool)
    orchestration.add_argument('--experiment', help="Name of mlflow experiment", default='Default')
    orchestration.add_argument('-R', '--create-new-run', help="Create as new MLFLow run, even if continuing", action='store_true')
    orchestration.add_argument('-B', '--base-path', default='.')
    orchestration.add_argument('--save-every', default=10, type=int)
    orchestration.add_argument('--st_checkpoint_dir', help="checkpoint dir for synetune", type=str, default=None)
    orchestration.add_argument('--use-mlflow', help="whether to use mlflow", action='store_true')
    orchestration.add_argument('--use-wandb', help="whether to use wandb", action='store_true')
    orchestration.add_argument('-f', '--warm-start-from', help='Warm start from this file')
    orchestration.add_argument('-c', '--continue-run', help='Whether to read the old config when warm starting', action='store_true')
    orchestration.add_argument('-s', '--load-strict', help='Whether to load the architecture strictly when warm starting', action='store_true')
    orchestration.add_argument('--restart-scheduler', help='Whether to restart the scheduler when warm starting', action='store_true')
    orchestration.add_argument('--detect-anomaly', help='Whether enable anomaly detection in pytorch. For debugging only.', action='store_true')
    orchestration.add_argument('--validate', type=str2bool, help='Whether to perform validation.', default=True)
    orchestration.add_argument('--progress-bar', type=str2bool, help='Whether to show a progress bar.', default=False)
    orchestration.add_argument('--wandb-overwrite', help='Whether to overwrite wandb runs.', action='store_true', default=False)
    orchestration.add_argument('--rl-validate-enabled', type=str2bool, help='Enable gym out-of-prior validation for rlpfn.')
    orchestration.add_argument('--rl-validate-envs', type=str, help='Comma-separated gym env list for rlpfn validation.')
    orchestration.add_argument('--rl-validate-episodes', type=int, help='Episodes per env during rlpfn validation.')
    orchestration.add_argument('--rl-validate-max-steps', type=int, help='Max steps per episode during rlpfn validation.')
    orchestration.add_argument('--rl-validate-action-candidates', type=int, help='Number of sampled continuous actions per step.')
    orchestration.add_argument('--rl-validate-seed', type=int, help='Base random seed for rlpfn validation.')

    if model_type == 'rlpfn':
        orchestration.set_defaults(
            rl_validate_enabled=True,
            rl_validate_envs=",".join(RLPFN_DEFAULT_OOP_ENVS),
            rl_validate_episodes=3,
            rl_validate_max_steps=1000,
            rl_validate_action_candidates=16,
            rl_validate_seed=1,
        )
    else:
        orchestration.set_defaults(
            rl_validate_enabled=False,
            rl_validate_envs=",".join(RLPFN_DEFAULT_OOP_ENVS),
            rl_validate_episodes=3,
            rl_validate_max_steps=1000,
            rl_validate_action_candidates=16,
            rl_validate_seed=1,
        )

    # orchestration options are not part of the default config
    return parser
