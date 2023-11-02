import argparse

NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale"]

def add_args(parser):
    # model architecture options
    parser.add_argument('--f_dim', type=int, default=28,
                        help='Number of feature input dimensions (28 for teamA(XY), teamB(XY), ball(XY), ball status')
    parser.add_argument('--c_dim', type=int, default=18,
                        help='Number of condition input dimensions (12 for teamA(XY), ball(XY)')
    parser.add_argument('--t_dim', type=int, default=120,
                        help='Number of time input dimensions (120 for time length)')
    parser.add_argument('--z_dim', type=int, default=384,
                        help='Number of latent code dimensions')
    parser.add_argument('--encode_depth', type=int, default=1)
    parser.add_argument('--encode_patch_num', type=int, default=120)
    parser.add_argument('--generate_depth', type=int, default=4)
    parser.add_argument('--generate_patch_num', type=int, default=120)
    parser.add_argument('--extract_depth', type=int, default=1)
    parser.add_argument('--extract_patch_num', type=int, default=24)
    parser.add_argument('--critic_depth', type=int, default=4)
    parser.add_argument('--critic_patch_num', type=int, default=24)
    parser.add_argument('--critic_iteration', type=int, default=1)
    parser.add_argument('--critic_pretrain_epochs', type=int, default=10)
    parser.add_argument('--critic_pretrain_iteration', type=int, default=1)
    parser.add_argument('--latent_hidden_dim', type=int, default=384, help='Hidden dim of stacked CNFs.')
    parser.add_argument('--latent_num_blocks', type=int, default=4, help='Number of stacked CNFs.')
    parser.add_argument('--layer_type', type=str, default="concatsquash", choices=LAYERS)
    parser.add_argument('--time_length', type=float, default=1)
    parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
    parser.add_argument('--nonlinearity', type=str, default='tanh', choices=NONLINEARITIES)
    parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)
    
    # training options
    parser.add_argument('--d_lr', type=float, default=1e-6)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam.')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum for SGD, ')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs for training (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for initializing training. ')
    parser.add_argument('--kl_weight', type=float, default=1e-4)
    parser.add_argument('--recon_weight', type=float, default=1e-0)
    parser.add_argument('--sample_weight', type=float, default=1e-2)
    parser.add_argument('--feature_weight', type=float, default=1e-1)
    parser.add_argument('--relation_weight', type=float, default=2.)
    parser.add_argument('--speed_weight', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, default=None,
                        help='Type of learning rate schedule(exponential, step, linear)')
    parser.add_argument('--exp_decay', type=float, default=1.,
                        help='Learning rate schedule exponential decay rate')
    parser.add_argument('--exp_decay_freq', type=int, default=1,
                        help='Learning rate exponential decay frequency')
    
    # logging and saving frequency
    parser.add_argument('--log_name', type=str, default='log', help="Name for the log dir")
    parser.add_argument('--val_freq', type=int, default=20)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=20)
    
    # validation options
    parser.add_argument('--no_validation', action='store_true',
                        help='Whether to disable validation altogether.')
    parser.add_argument('--save_val_results', action='store_true',
                        help='Whether to save the validation results.')
    
    # distributed training
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')
    
    # data path
    parser.add_argument('--data_path', type=str, default='data',
                        help='string, path of target data')
    parser.add_argument('--model_path', type=str, default='checkpoints/',
                        help='string, path of target pretrained model')
    
    return parser

def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='Flow-based Point Cloud Generation Experiment')
    parser = add_args(parser)
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
