import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_loader.msrs_data import MSRS_data
from models.common import clamp

from models.vir_branch import Fusion
from models.cls_model import LoraCLIP
from models.common import YCrCb2RGB

import time
import traceback

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def tensor_to_pil(tensor):

    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:          # (B,C,H,W) -> (C,H,W)
        tensor = tensor[0]
    if tensor.ndim == 2:          # (H,W) -> (1,H,W)
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:      # single channel -> 3‑channel repeat
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor.float().clamp(0, 1)
    return transforms.ToPILImage()(tensor)

if __name__ == '__main__':
    dataset = 'test_data/visual'
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--dataset_path', metavar='DIR', default=f'{dataset}',
                        help='path to dataset (default: imagenet)')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default=f'{dataset}/output')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    test_dataset = MSRS_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.arch == 'fusion_model':
        model = Fusion().cuda()
        model.load_state_dict(torch.load('runs/F.pth'))
        model.eval()

        #cls_model = LoraCLIP(num_classes=4)
        cls_model = torch.load('./best_cls.pth')
        cls_model.cuda()
        cls_model.eval()

        print("=" * 50)
        print(f"Fusion model Parameters: {count_parameters(model):,}")
        print(f"CLS model Parameters: {count_parameters(cls_model):,}")
        print("=" * 50)
        test_tqdm = tqdm(test_loader, total=len(test_loader))

        with torch.no_grad():
            for vis_image_y, cr, cb, inf_image, vis_image_clip, inf_image_clip, name in test_tqdm:
                try:
                    start_time = time.time()

                    vis_image = vis_image_y.cuda()
                    inf_image = inf_image.cuda()
                    cr, cb = cr.cuda(), cb.cuda()

                    # expand infrared for CLIP
                    vis_image_clip, inf_image_clip = vis_image_clip.cuda(), inf_image_clip.cuda()
                    if inf_image_clip.shape[1] == 1:
                        inf_image_clip = inf_image_clip.repeat(1, 3, 1, 1)
                    if vis_image_clip.shape[1] == 1:
                        vis_image_clip = vis_image_clip.repeat(1, 3, 1, 1)

                    _, feat_vis = cls_model(vis_image_clip)
                    _, feat_inf = cls_model(inf_image_clip)

                    vi_r, ir_r, fx_vi_branch, _ = model(vis_image, inf_image, feat_vis * feat_inf)

                    fused_rgb1 = tensor_to_pil(YCrCb2RGB(clamp(vi_r)[0], cr[0], cb[0]))
                    fused_rgb2 = tensor_to_pil(YCrCb2RGB(clamp(fx_vi_branch)[0], cr[0], cb[0]))
                    ir_rgb     = tensor_to_pil(clamp(ir_r)[0])

                    save_path1 = args.save_path + '_vi'
                    save_path2 = args.save_path + '_ir'
                    save_path3 = args.save_path + '_fx'
                    for p in (save_path1, save_path2, save_path3):
                        os.makedirs(p, exist_ok=True)

                    fused_rgb1.save(f'{save_path1}/{name[0]}')
                    fused_rgb2.save(f'{save_path3}/{name[0]}')
                    ir_rgb.save   (f'{save_path2}/{name[0]}')

                    elapsed_ms = (time.time() - start_time) * 1000
                    print(f"{name[0]} Processing time: {elapsed_ms:.2f} ms")
                except Exception as e:
                    print(f"{name[0]} Wrong: {e}")
                    traceback.print_exc()
                    continue