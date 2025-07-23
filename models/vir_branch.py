import torch
import torch.nn as nn
from .MItransformer import MMoEFeedForward
step_dim = 16
from .fusion_model import Downsample, Upsample
class MoEBlock(nn.Module):
    def __init__(self, dim=16, out_channels=32, num_experts=8, top_k=2):
        super().__init__()
        self.block = nn.Sequential(
            MMoEFeedForward(dim=dim, num_experts=num_experts, top_k=top_k),
            
            Downsample(n_feat=dim, out_channels=out_channels) if dim<out_channels else Upsample(n_feat=dim, out_channels=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x, text_feature, task_id=1):
        x, mi_loss = self.block[0](x, text_feature, task_id=task_id)
        x = self.block[1](x)
        x = self.block[2](x)
        return x, mi_loss

class vir_branch_encode(nn.Module):
    def __init__(self):
        super(vir_branch_encode, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=step_dim, stride=1, padding=1, kernel_size=3)

        self.moe_blocks_encode = nn.ModuleList([MoEBlock(dim=(i+1)*step_dim, out_channels=(i+2)*step_dim) for i in range(4)])

    def forward(self, x, text_feature, task_id):
        
        x = self.conv(x)
        x1, mi_loss1 = self.moe_blocks_encode[0](x, text_feature,task_id)
        x2, mi_loss2 = self.moe_blocks_encode[1](x1, text_feature,task_id)
        x3, mi_loss3 = self.moe_blocks_encode[2](x2, text_feature,task_id)
        x4, mi_loss4 = self.moe_blocks_encode[3](x3, text_feature,task_id)
        return [x1, x2, x3, x4], (mi_loss1+mi_loss2+mi_loss3+mi_loss4)

from .fusion_model import IndependentSpatialGatedFusionBlock, GatedFusionBlock
class vir_branch_decode(nn.Module):
    def __init__(self):
        super(vir_branch_decode, self).__init__()
        self.moe_blocks_decode = nn.ModuleList(
            [MoEBlock(dim=((4-i) + 1) * step_dim, out_channels=(4-i) * step_dim) for i in range(4)])
        self.conv = nn.Conv2d(in_channels=step_dim, out_channels=1, stride=1, padding=1, kernel_size=3)

    def forward(self, x_encode, text_feature):
        y1, mi_loss1 = self.moe_blocks_decode[0](x_encode[3], text_feature)
        y2, mi_loss2 = self.moe_blocks_decode[1](x_encode[2]+y1, text_feature)
        y3, mi_loss3 = self.moe_blocks_decode[2](x_encode[1]+y2, text_feature)
        y4, mi_loss4 = self.moe_blocks_decode[3](x_encode[0]+y3, text_feature)
        img = self.conv(y4)
        return img, (mi_loss1+mi_loss2+mi_loss3+mi_loss4)

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.encode = vir_branch_encode()
        self.decode_vi = vir_branch_decode()
        self.decode_ir = vir_branch_decode()
        self.decode_fi = vir_branch_decode()
        
        self.fusion_gata = nn.ModuleList([GatedFusionBlock(channels=(i+2)*step_dim) for i in range(4)])



    def forward(self, vi, ir, text_feature):
        vi_x, mi_loss_vi_encode = self.encode(vi, text_feature, task_id=0)
        ir_x, mi_loss_ir_encode = self.encode(ir, text_feature, task_id=1)
        vi_r, mi_loss_vr_decode = self.decode_vi(vi_x, text_feature)
        ir_r, mi_loss_ir_decode = self.decode_ir(ir_x, text_feature)

        fx = []
        for vx, ix, fusion_gata in zip(vi_x, ir_x, self.fusion_gata):
            fx.append(fusion_gata(vx, ix))


        fx_vi_branch, mi_f_vi_branch = self.decode_fi(fx, text_feature)
        
        return vi_r, ir_r, fx_vi_branch, (mi_loss_vi_encode+mi_loss_ir_encode+mi_loss_vr_decode+mi_loss_ir_decode+mi_f_vi_branch)





if __name__ == '__main__':
    model = Fusion()
    img = torch.randn(1,1, 64, 64)
    text_feature = torch.randn(1,512)
    for i in model(img, img, text_feature):
        print(i.shape,"------------>OK!")
