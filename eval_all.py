import argparse
import os
import torch
import numpy as np
import warnings
import csv

from data.datasets_all import TestDataset
from engine_finetune_all import evaluate

######## import models ########
import models.AIDE_CHR as AIDE
import models.PMIL as PMIL
import models.PMIL_SRM as PMIL_SRM
import utils
###############################
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Start Evaluation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model', default='PMIL', type=str)
    parser.add_argument('--isTrain', default=False, type=utils.str2bool, help='Whether to train or evaluate')
    parser.add_argument('--resnet_path', type=str, default='/raid5/chr/AIGCD/AIDE/results/cnnspot-progan-res50-rgb/checkpoint-3.pth')
    parser.add_argument('--data_path', type=str, default='/raid0/chr/AIGCDetectBenchmark/AIGCDetect_testset/test')
    parser.add_argument('--eval_data_path', type=str, default='/raid5/chr/AIGCD/AIGCDetectBenchmark/AIGCDetect_testset/test')
    parser.add_argument('--output_dir', default='/raid5/chr/AIGCD/AIDE/eval_results', type=str)
    parser.add_argument('--csv_file_name', default='cnnspot-progan-rgb-e3',type=str, help='csv file name')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', type=utils.str2bool, default=False)
    parser.add_argument('--is_anyres', action='store_true', default=False,help='disable pin memory')
    parser.add_argument('--convnext_path', type=str, default=None)
    parser.add_argument('--select_data_list',nargs='+',
                    default=['stable_diffusion_v_1_4'], 
                    help='List of selected data (default: ["default_data"])')
    parser.add_argument('--data_mode',default='cnnspot',type=str,help='选择数据增广的模式')
    parser.add_argument('--bucket_sampler', default=False,type=bool,help='whether to use bucket sampler')
    parser.add_argument('--mil_eval_mode', default='none', type=str, help='whether to use mil model')
    parser.add_argument('--no_crop', action='store_true', default=False, help='disable crop')

    return parser

def main(args):
    print("Running AIDE Evaluation")
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(321)
    np.random.seed(321)

    # Load model
    if args.model == 'AIDE':
        model = AIDE.__dict__[args.model](resnet_path=args.resnet_path,convnext_path=args.convnext_path)
    elif args.model == 'PMIL':
        model = PMIL.__dict__[args.model](resnet_path=args.resnet_path)
    elif args.model == 'PMIL_SRM':
        model = PMIL_SRM.__dict__[args.model](resnet_path=args.resnet_path)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    ##############################################################
    model.to(device)
    model.eval()

    # Define subfolders to evaluate
    vals = os.listdir(args.eval_data_path)
    if len(vals) >= 16:
        vals = ["Chameleon","progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "stylegan2", "whichfaceisreal",
                "ADM", "Glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5",
                "VQDM", "wukong", "DALLE2"]
        # vals = ["progan"]
        args.select_data_list = vals
        eval_data_path_root = args.eval_data_path
        if args.data_mode == 'only_eval':
            rows = [["{} model testing on...".format(args.resnet_path), '', ''], ['testset', 'accuracy', 'avg precision', 'Acc_MIL']]
        else:
            rows = [["{} model testing on...".format(args.resnet_path), '', ''], ['testset', 'accuracy', 'avg precision']]
        for val in vals:
            eval_path = os.path.join(eval_data_path_root, val)
            args.eval_data_path = eval_path
            dataset_val = TestDataset(is_train=False, args=args)

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                prefetch_factor=8 if args.num_workers > 0 else None,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            # Evaluate without passing a loss
            test_stats, acc, ap = evaluate(args,data_loader_val, model, device)
            print(f"Testset: {val} | Accuracy: {acc:.5f}, AP: {ap:.5f}")
            print("***********************************")
            if 'mil_acc' in test_stats:
                rows.append([val, acc, ap, test_stats['mil_acc']])
            else:
                rows.append([val, acc, ap])


        os.makedirs(args.output_dir, exist_ok=True)
        csv_name = os.path.join(args.output_dir, args.csv_file_name + '.csv')
        with open(csv_name, 'w') as f:
            csv.writer(f).writerows(rows)

    elif len(vals) == 8:
        select_data_list = ["Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5",
                "ADM", "glide", "wukong", "VQDM", "BigGAN"]
        args.select_data_list = vals
        dataset_val = TestDataset(is_train=False, args=args,select_data_list=select_data_list)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor= 8 if args.num_workers > 0 else None,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # Evaluate without passing a loss
        test_stats, acc, ap = evaluate(args,data_loader_val, model, device, criterion=None)

    else:
        print("Unknown dataset structure. Using all subfolders in path as eval targets.")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Start Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)