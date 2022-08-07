import argparse


parser = argparse.ArgumentParser(description='VDSR')

# dataset configs
parser.add_argument('--train_dataset_dir', default=None)
parser.add_argument('--test_dataset_dir', default=None)
parser.add_argument('--img_size', type=int, nargs='+', default=(160, 160), help='Image size')
parser.add_argument('--lr_scale', type=int, default=4, help='low resolution scale')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')

# train configs
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
parser.add_argument('--reset', type=bool, default=False, help='reset save_dir')
parser.add_argument('--save_dir', type=str, default='save_dir', help='Directory to save log, arguments, models')
parser.add_argument('--log_file_name', type=str, default='VDSR.log', help='Log file name')
parser.add_argument('--logger_name', type=str, default='VDSR', help='Logger name')

# device configs
parser.add_argument('--cpu', type=bool, default=False, help='Use CPU')
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')

# model configs
parser.add_argument('--img_channels', type=int, default=3, help='Number of channels of input image')
parser.add_argument('--res_channels', type=int, default=64, help='Number of channels of residual block')
parser.add_argument('--num_res_blocks', type=int, default=18, help='Number of residual blocks')

# training parameters
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-1, help='Learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
parser.add_argument('--lr_decay_step', type=int, default=20, help='Learning rate decay step')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--print_every', type=int, default=99999, help='Print every')
parser.add_argument('--save_every', type=int, default=99999, help='Save every')
parser.add_argument('--val_every', type=int, default=99999, help='Validate every')

# eval & test configs
parser.add_argument('--eval', type=bool, default=False, help='Eval mode')
parser.add_argument('--test', type=bool, default=False, help='Test mode')
parser.add_argument('--eval_save_results', type=bool, default=False, help='Save results on Eval')
parser.add_argument('--model_path', type=str, default=None, help='Model path')
parser.add_argument('--lr_path', type=str, default='./test/demo/lr/lr.png', help='LR path')

args = parser.parse_args()