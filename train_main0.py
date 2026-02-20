import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from data_loader.msrs_data1 import MSRS_data
from models.common import gradient, clamp
from models.vir_branch import Fusion
from models.cls_model import LoraCLIP  # 导入LoraCLIP用于创建不同rank的模型
from pytorch_msssim import ssim
from models.common import RGB2YCrCb
criterion = nn.CrossEntropyLoss()

# def init_seeds(seed=0):
#     import torch.backends.cudnn as cudnn
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

if __name__ == '__main__':
    # ========== 可配置参数 ==========
    lora_rank = 8  # 在这里修改 LoRA Rank: 0, 2, 4, 8, 16
    # ================================
    
    train_dataset_path = 'datasets1'
    train_dataset_path_hav = 'Havard-noise'
    batch_size = 1
    workers = 1
    lr = 0.0002
    epochs = 300
    save_path = f'runs/rank_{lora_rank}'  # 根据rank创建不同的保存目录

    train_dataset = MSRS_data(train_dataset_path, task=0)
    train_dataset_hav = MSRS_data(train_dataset_path_hav, task=1)
    train_dataset_hav_noi = MSRS_data(train_dataset_path_hav, task=2)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    train_loader_hav = DataLoader(
        train_dataset_hav, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )
    train_loader_hav_noi = DataLoader(
        train_dataset_hav_noi, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Fusion()
    for param in model.decode_fi.parameters():
        param.requires_grad = False
    for param in model.fusion_gata.parameters():
        param.requires_grad = False
    model = model.cuda()
    #model.load_state_dict(torch.load('runs/F.pth'))
    
    # ========== 创建CLIP模型（支持不同LoRA rank）============
    cls_model_path = './best_cls.pth'
    
    # 如果 rank=4 且预训练模型存在，直接加载
    if lora_rank == 4 and os.path.exists(cls_model_path):
        print(f"加载预训练CLIP模型 (rank={lora_rank})")
        cls_model = torch.load(cls_model_path, weights_only=False)
        cls_model.cuda()
        cls_model.eval()
    else:
        # 创建新模型（指定rank）
        print(f"创建新的CLIP模型 (rank={lora_rank})")
        cls_model = LoraCLIP(num_classes=9, r=lora_rank, pretrained=True)
        cls_model.cuda()
        cls_model.eval()
        
        # 如果 rank=4，尝试加载预训练权重（兼容性加载）
        if lora_rank == 4 and os.path.exists(cls_model_path):
            try:
                pretrained_model = torch.load(cls_model_path, weights_only=False)
                cls_model.load_state_dict(pretrained_model.state_dict(), strict=False)
                print(f"已加载预训练CLIP模型权重 (rank={lora_rank})")
            except Exception as e:
                print(f"无法加载预训练权重，使用新初始化的模型: {e}")
    # ========================================================

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        if epoch < epochs // 2:
            lr = lr
        else:
            lr = lr * (epochs - epoch) / (epochs - epochs // 2)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        train_tqdm = tqdm(train_loader, total=len(train_loader))
        for _ in range(2):
            for vis_image, inf_image, vis_gt, inf_gt, [vis_image_clip, inf_image_clip], name in train_tqdm:
                vis_image = vis_image.cuda()
                inf_image = inf_image.cuda()
                vis_gt = vis_gt.cuda()
                inf_gt = inf_gt.cuda()
                vis_image_clip = vis_image_clip.cuda()
                inf_image_clip = inf_image_clip.cuda()
                _, c, _, _ = inf_image_clip.shape
                if c==1:
                    inf_image_clip = torch.cat([inf_image_clip]*3, dim=1)
                _, feature_vis = cls_model(vis_image_clip)
                _, feature_inf = cls_model(inf_image_clip)
                feature = feature_vis * feature_inf

                optimizer.zero_grad()

                vi_r, ir_r, fx_vi_branch, loss_mi = model(vis_image, inf_image, feature)
                #print(vi_r.shape, ir_r.shape, vis_gt.shape, vis_image.shape)
                # 先写重建损失
                loss_re_pix = F.l1_loss(vi_r, vis_gt) + F.l1_loss(ir_r, inf_gt)
                loss_re_ssim = 2-ssim(vi_r, vis_gt)-ssim(ir_r, inf_gt)
                # 融合损失

                # 总损失
                loss = 100 *loss_re_ssim + 10 * loss_re_pix + 0.01*loss_mi

                loss.backward()
                optimizer.step()

                train_tqdm.set_postfix(epoch=epoch,
                                       loss=loss.item(),
                                       loss_re_pix=10*loss_re_pix.item(),
                                       loss_re_ssim=100*loss_re_ssim.item(),
                                       loss_mi=loss_mi.item(),
                                       task=0
                                       )
        train_tqdm1 = tqdm(train_loader_hav, total=len(train_loader_hav))

        for vis_image, inf_image, vis_gt, inf_gt, [vis_image_clip, inf_image_clip], name in train_tqdm1:
            vis_image = vis_image.cuda()
            inf_image = inf_image.cuda()
            vis_gt = vis_gt.cuda()
            inf_gt = inf_gt.cuda()
            vis_image_clip = vis_image_clip.cuda()
            inf_image_clip = inf_image_clip.cuda()
            _, c, _, _ = inf_image_clip.shape
            if c==1:
                inf_image_clip = torch.cat([inf_image_clip]*3, dim=1)
            _, c, _, _ = vis_image_clip.shape
            if c == 1:
                vis_image_clip = torch.cat([vis_image_clip] * 3, dim=1)
            _, feature_vis = cls_model(vis_image_clip)
            _, feature_inf = cls_model(inf_image_clip)
            feature = feature_vis * feature_inf

            optimizer.zero_grad()

            vi_r, ir_r, fx_vi_branch, loss_mi = model(vis_image, inf_image, feature)
            #print(vi_r.shape, ir_r.shape, vis_gt.shape, vis_image.shape)
            # 先写重建损失
            loss_re_pix = F.l1_loss(vi_r, vis_gt) + F.l1_loss(ir_r, inf_gt)
            loss_re_ssim = 2-ssim(vi_r, vis_gt)-ssim(ir_r, inf_gt)
            # 融合损失

            # 总损失
            loss = 100 *loss_re_ssim + 10 * loss_re_pix + 0.01*loss_mi

            loss.backward()
            optimizer.step()

            train_tqdm1.set_postfix(epoch=epoch,
                                   loss=loss.item(),
                                   loss_re_pix=10*loss_re_pix.item(),
                                   loss_re_ssim=100*loss_re_ssim.item(),
                                   loss_mi=loss_mi.item(),
                                   task=1
                                   )
        # train_tqdm2 = tqdm(train_loader_hav_noi, total=len(train_loader_hav_noi))
        # for vis_image, inf_image, vis_gt, inf_gt, [vis_image_clip, inf_image_clip], name in train_tqdm2:
        #     vis_image = vis_image.cuda()
        #     inf_image = inf_image.cuda()
        #     vis_gt = vis_gt.cuda()
        #     inf_gt = inf_gt.cuda()
        #     vis_image_clip = vis_image_clip.cuda()
        #     inf_image_clip = inf_image_clip.cuda()
        #     _, c, _, _ = inf_image_clip.shape
        #     if c==1:
        #         inf_image_clip = torch.cat([inf_image_clip]*3, dim=1)
        #     _, c, _, _ = vis_image_clip.shape
        #     if c == 1:
        #         vis_image_clip = torch.cat([vis_image_clip] * 3, dim=1)
        #     _, feature_vis = cls_model(vis_image_clip)
        #     _, feature_inf = cls_model(inf_image_clip)
        #     feature = feature_vis * feature_inf
        #
        #     optimizer.zero_grad()
        #
        #     vi_r, ir_r, fx_vi_branch, loss_mi = model(vis_image, inf_image, feature)
        #     #print(vi_r.shape, ir_r.shape, vis_gt.shape, vis_image.shape)
        #     # 先写重建损失
        #     loss_re_pix = F.l1_loss(vi_r, vis_gt) + F.l1_loss(ir_r, inf_gt)
        #     loss_re_ssim = 2-ssim(vi_r, vis_gt)-ssim(ir_r, inf_gt)
        #     # 融合损失
        #
        #     # 总损失
        #     loss = 100 *loss_re_ssim + 10 * loss_re_pix + 0.01*loss_mi
        #
        #     loss.backward()
        #     optimizer.step()
        #
        #     train_tqdm2.set_postfix(epoch=epoch,
        #                            loss=loss.item(),
        #                            loss_re_pix=10*loss_re_pix.item(),
        #                            loss_re_ssim=100*loss_re_ssim.item(),
        #                            loss_mi=loss_mi.item(),
        #                            task=2
        #                            )

        torch.save(model.state_dict(), f'{save_path}/F_base.pth')
