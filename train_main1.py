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
    train_dataset_path = 'datasets1'
    train_dataset_path_hav = 'Havard-noise'
    batch_size = 1
    workers = 1
    lr = 0.0001
    epochs = 500
    save_path = 'runs'

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
    model.load_state_dict(torch.load('runs/F_base.pth'))
    for param in model.encode.parameters():
        param.requires_grad = False
    for param in model.decode_vi.parameters():
        param.requires_grad = False
    for param in model.decode_ir.parameters():
        param.requires_grad = False
    model = model.cuda()
    cls_model = torch.load('./best_cls.pth')
    cls_model.cuda()
    cls_model.eval()

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
        for i in range(2):
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
                #loss_re = F.l1_loss(vi_r, vis_gt) + F.l1_loss(ir_r, inf_gt)
                # 融合损失
                #loss_fi_1 = F.l1_loss(gradient(fx_ir_branch), torch.max(gradient(vis_gt), gradient((inf_gt))))
                loss_fi_gard = F.l1_loss(gradient(fx_vi_branch), torch.max(gradient(vis_gt), gradient(inf_gt)))
                loss_fi_ssim = 2 - ssim(fx_vi_branch, vis_gt) - ssim(fx_vi_branch, inf_gt)
                loss_fi_pix = F.l1_loss(fx_vi_branch, torch.max(vis_gt, inf_gt))


                loss_fi = 50*loss_fi_gard + 0*loss_fi_ssim + 20*loss_fi_pix

                # 总损失
                #loss = 10 *loss_re + loss_fi + 0.01 * loss_mi
                loss = loss_fi + 0.001 * loss_mi

                loss.backward()
                optimizer.step()

                train_tqdm.set_postfix(epoch=epoch,
                                       #loss_re=10*loss_re.item(),
                                       #loss_fi_1=50*loss_fi_1.item(),
                                       #loss_fi_2=50*loss_fi_2.item(),
                                       loss_fi=loss_fi.item(),
                                       #loss_fi_3=loss_fi_3.item(),
                                       loss_mi=loss_mi.item(),
                                       loss_total=loss.item())
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
            #loss_re = F.l1_loss(vi_r, vis_gt) + F.l1_loss(ir_r, inf_gt)
            # 融合损失
            #loss_fi_1 = F.l1_loss(gradient(fx_ir_branch), torch.max(gradient(vis_gt), gradient((inf_gt))))
            loss_fi_gard = F.l1_loss(gradient(fx_vi_branch), torch.max(gradient(vis_gt), gradient(inf_gt)))
            loss_fi_ssim = 2 - ssim(fx_vi_branch, vis_gt) - ssim(fx_vi_branch, inf_gt)
            loss_fi_pix = F.l1_loss(fx_vi_branch, torch.max(vis_gt, inf_gt))


            loss_fi = 50*loss_fi_gard + 10*loss_fi_ssim + 20*loss_fi_pix

            # 总损失
            #loss = 10 *loss_re + loss_fi + 0.01 * loss_mi
            loss = loss_fi + 0.01 * loss_mi

            loss.backward()
            optimizer.step()

            train_tqdm1.set_postfix(epoch=epoch,
                                   #loss_re=10*loss_re.item(),
                                   #loss_fi_1=50*loss_fi_1.item(),
                                   #loss_fi_2=50*loss_fi_2.item(),
                                   loss_fi=loss_fi.item(),
                                   #loss_fi_3=loss_fi_3.item(),
                                   loss_mi=loss_mi.item(),
                                   loss_total=loss.item())


        torch.save(model.state_dict(), f'{save_path}/F.pth')
