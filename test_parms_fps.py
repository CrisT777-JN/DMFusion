
import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

# --- Checkpoint 加载函数 ---

if __name__ == '__main__':
    dataset = 'test_data/LLVIP'
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--dataset_path', metavar='DIR', default=f'{dataset}',
                        help='path to dataset (default: imagenet)')  # test data

    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default=f'{dataset}/ours')  # output in ours_ir
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

        cls_model = torch.load('./best_cls.pth')
        cls_model.cuda()
        cls_model.eval()

        print("=" * 50)
        print(f"Fusion model parameters: {count_parameters(model):,}")
        print("=" * 50)
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        with torch.no_grad():
            for vis_image_y, cr, cb, inf_image, vis_image_clip, inf_image_clip, name in test_tqdm:
                try:
                    
                    start_time = time.time()  
                    vis_image = vis_image_y.cuda()
                    inf_image = inf_image.cuda()
                    cr, cb = cr.cuda(), cb.cuda()
                    
                    vis_image_clip = vis_image_clip.cuda()
                    inf_image_clip = inf_image_clip.cuda()
                    _, c, _, _ = inf_image_clip.shape
                    if c==1:
                        inf_image_clip = torch.cat([inf_image_clip]*3, dim=1)
                    _, c, _, _ = vis_image_clip.shape
                    _, feature_vis = cls_model(vis_image_clip)
                    _, feature_inf = cls_model(inf_image_clip)

                    vi_r, ir_r, fx_vi_branch, _= model(vis_image, inf_image, feature_vis*feature_inf)

                    fused_image1 = clamp(vi_r)
                    fused_image1 = YCrCb2RGB(fused_image1[0], cr[0], cb[0])

                    fused_image2 = clamp(fx_vi_branch)
                    fused_image2 = YCrCb2RGB(fused_image2[0], cr[0], cb[0])

                    rgb_fused_image1 = transforms.ToPILImage()(fused_image1)
                    rgb_fused_image2 = transforms.ToPILImage()(fused_image2)
                    save_path1 = args.save_path + '_vi'
                    save_path2 = args.save_path + '_ir'
                    if not os.path.exists(save_path1):
                        os.mkdir(save_path1)
                    if not os.path.exists(save_path2):
                        os.mkdir(save_path2)
                    rgb_fused_image1.save(f'{save_path1}/{name[0]}')
                    rgb_fused_image2.save(f'{save_path2}/{name[0]}')
                    end_time = time.time()  
                    elapsed_time = (end_time - start_time) * 1000  
                    print(f"{name[0]} Proceeding time: {elapsed_time:.2f} ms")
                except:
                    print(name[0],'------------------->shape mismatch')