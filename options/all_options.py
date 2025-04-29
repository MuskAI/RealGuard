import argparse
from utils import str2bool, remap_checkpoint_keys
def get_args_parser():
    parser = argparse.ArgumentParser('Resnet fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=10, type=int)
    # added by haoran 
    # 选择数据预处理的方法：
    # 1. 默认CNNSpot的方法 "cnnspot"
    # 2. 更强的CNNSpot增广方式
    # 3. 为MIL而使用的增广方式 "mil"
    parser.add_argument('--model', default='PMIL', type=str, metavar='MODEL',
                        help='Name of model to train') # AIDE or PMIL
    parser.add_argument('--data_mode',default='cnnspot',type=str,help='选择数据增广的模式')
    # 选择是否开启mil 模式，也就是是否使用MIL Loss
    parser.add_argument('--loss_mode',default='cnnspot',type=str,help='选择loss模式')
    # parser.add_argument('--mil_mode', default=False, type=bool, help='whether to use mil model')
    parser.add_argument('--data_path', default='/raid0/chr/AIGCD/Datasets/GenImage', type=str,
                        help='dataset path')
    parser.add_argument('--select_data_list',nargs='+',
                        default=['stable_diffusion_v_1_4'], 
                        help='List of selected data (default: ["default_data"])')
    parser.add_argument('--eval_data_path', default='/raid0/chr/AIGCD/Datasets/GenImage', type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--output_dir', default='/raid5/chr/AIGCD/AIDE/results/cnnspot-progan',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--bucket_sampler', default=False,type=bool,help='whether to use bucket sampler')
    parser.add_argument('--mil_eval_mode', default='', type=str, help='whether to use mil model')
    parser.add_argument('--is_earlystop', default=False, type=bool, help='whether to use earlystop')
    #### Testing ######
    parser.add_argument('--no_crop', action='store_true', help='no crop')
    parser.add_argument('--no_resize', action='store_true', help='no resize')
    #############################################################################
    parser.add_argument('--prefetch_factor', default=32,type=int,help='prefetch factor')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=100, type=int)
    #####################
    parser.add_argument('--update_freq', default=1, type=int, help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--resnet_path', default='/raid5/chr/AIGCD/AIDE/pretrained_ckpts/resnet50-19c8e357.pth', type=str, metavar='MODEL',
                        help='Path of resnet model')
    parser.add_argument('--convnext_path', default=None, type=str, metavar='MODEL',
                        help='Path of ConvNeXt of model ')
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')    
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                       help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=0.001, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    

    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int) 
    parser.add_argument('--local-rank', default=-1, type=int) # added by haoran
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use apex AMP (Automatic Mixed Precision) or not")
    

    return parser
