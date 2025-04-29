import argparse
import os
import torch
import numpy as np
import warnings
import csv

from data.datasets_only_mileval import TestDataset
from engine_finetune_mil import evaluate

import models.AIDE_CHR as AIDE
import utils

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('AIDE Evaluation', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model', default='AIDE', type=str)
    parser.add_argument('--isTrain', default=False, type=utils.str2bool, help='Whether to train or evaluate')
    parser.add_argument('--resnet_path', type=str, default='/raid5/chr/AIGCD/AIDE/results/AIDE-Original-progan/checkpoint-8.pth')
    parser.add_argument('--data_path', type=str, default='/raid0/chr/AIGCDetectBenchmark/AIGCDetect_testset/test')
    parser.add_argument('--eval_data_path', type=str, default='/raid5/chr/AIGCD/AIGCDetectBenchmark/AIGCDetect_testset/test')
    parser.add_argument('--output_file', type=str, default='AIDE-e8-ONLY-MILEVAL.csv')
    parser.add_argument('--output_dir', default='/raid5/chr/AIGCD/AIDE/eval_results', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', type=utils.str2bool, default=False)
    parser.add_argument('--convnext_path', type=str, default=None)
    parser.add_argument('--is_anyres', action='store_true', default=True,help='disable pin memory')

    return parser

def main(args):
    print("Running AIDE Evaluation")
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(321)
    np.random.seed(321)

    # Load model
    model = AIDE.__dict__[args.model](resnet_path=args.resnet_path,convnext_path=args.convnext_path)
    model.to(device)
    model.eval()

    # Define subfolders to evaluate
    vals = os.listdir(args.eval_data_path)
    if len(vals) >= 16:
        vals = ["Chameleon","progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "stylegan2", "whichfaceisreal",
                "ADM", "Glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5",
                "VQDM", "wukong", "DALLE2"]
        # vals = ["progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "stylegan2", "whichfaceisreal",
        #         "ADM", "Glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5",
        #         "VQDM", "wukong", "DALLE2"]       
        
        eval_data_path_root = args.eval_data_path
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
                pin_memory=args.pin_mem,
                prefetch_factor=8,
                drop_last=False
            )

            # Evaluate without passing a loss
            test_stats, acc, ap = evaluate(data_loader_val, model, device, criterion=None)
            print(f"Testset: {val} | Accuracy: {acc:.5f}, AP: {ap:.5f}")
            print("***********************************")
            rows.append([val, acc, ap])

        os.makedirs(args.output_dir, exist_ok=True)
        # csv_name = os.path.join(args.output_dir, f'{os.path.basename(args.resume)}_eval_results.csv')
        csv_name = os.path.join(args.output_dir, args.output_file)
        with open(csv_name, 'w') as f:
            csv.writer(f).writerows(rows)

    elif len(vals) == 8:
        # select_data_list = ["Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5",
        #         "ADM", "glide", "wukong", "VQDM", "BigGAN"]
        select_data_list = ["stable_diffusion_v_1_4"]
        dataset_val = TestDataset(is_train=False, args=args,select_data_list=select_data_list)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # Evaluate without passing a loss
        test_stats, acc, ap = evaluate(data_loader_val, model, device, criterion=None)

    else:
        print("Unknown dataset structure. Using all subfolders in path as eval targets.")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('AIDE Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)