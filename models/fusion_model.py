import torch
from models.transformer import MTransformerBlock, MoEFeedForward
import torch.nn as nn
import torch.nn.functional as F

class IndependentSpatialGatedFusionBlock(nn.Module):
    def __init__(self, channels):
        super(IndependentSpatialGatedFusionBlock, self).__init__()
        reduction = max(4, channels // 8)
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            # --- Change Start ---
            nn.Conv2d(channels // reduction, channels*2, 1, bias=False),
            # --- Change End ---
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, v, i):
        combined = torch.cat([v, i], dim=1)
        gates = self.attention(combined) # (B, 2*C, H, W)
        # --- Change Start ---
        gate_v, gate_i = torch.split(gates,self.channels, dim=1) # (B, C, H, W) each
        #print(gate_v, gate_i)
        gated_fusion = gate_v * v + gate_i * i
        gated_fusion = self.final_conv(gated_fusion) # 可选
        # --- Change End ---
        return gated_fusion

class GatedFusionBlock(nn.Module):
    
    def __init__(self, channels):
        super(GatedFusionBlock, self).__init__()
        reduction = max(4, channels // 8)  
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
           
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
           
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, v, i):
        
        combined = torch.cat([v, i], dim=1)
        gate = self.attention(combined) # shape: (B, C, 1, 1)
        
        gated_fusion = v + gate * i
        return gated_fusion




class Downsample(nn.Module):
    def __init__(self, n_feat, out_channels):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),  
            nn.Conv2d(in_channels=n_feat * 4, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=False)
        )


    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat,out_channels):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2),
                                  nn.Conv2d(in_channels=n_feat//4, out_channels=out_channels, kernel_size=3, stride=1,
                                            padding=1,
                                            bias=False)
                                  )

    def forward(self, x):
        return self.body(x)

from models.MItransformer import HeterogeneousTransformerBlock
class encoder(nn.Module):
    def __init__(self, d_text=512):
        super(encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, stride=1, padding=1, kernel_size=3)
        self.moe_vit_block1 = HeterogeneousTransformerBlock(dim=16, d_text=d_text, num_experts=8, top_k=2)
        self.down1 = Downsample(n_feat=16)
        self.moe_vit_block2 = HeterogeneousTransformerBlock(dim=32, d_text=d_text, num_experts=8, top_k=2)
        self.down2 = Downsample(n_feat=32)
        self.moe_vit_block3 = HeterogeneousTransformerBlock(dim=64, d_text=d_text, num_experts=8, top_k=2)
        self.down3 = Downsample(n_feat=64)
        self.moe_vit_block4 = HeterogeneousTransformerBlock(dim=128, d_text=d_text, num_experts=8, top_k=2)
        self.down4 = Downsample(n_feat=128)

    def forward(self, x, route_feature, task_id):
        # Initialize aggregated aux losses dictionary
        device = x.device
        total_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                            'mi_loss': torch.tensor(0.0, device=device)}

        x1_ = self.conv(x)
        x1_moe, aux_losses1 = self.moe_vit_block1(x1_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses1.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses1.get('mi_loss', 0.0)
        x2_ = self.down1(x1_moe)

        x2_moe, aux_losses2 = self.moe_vit_block2(x2_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses2.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses2.get('mi_loss', 0.0)
        x3_ = self.down2(x2_moe)

        x3_moe, aux_losses3 = self.moe_vit_block3(x3_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses3.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses3.get('mi_loss', 0.0)
        x4_ = self.down3(x3_moe)

        x4_moe, aux_losses4 = self.moe_vit_block4(x4_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses4.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses4.get('mi_loss', 0.0)
        x5_ = self.down4(x4_moe) # Bottleneck

        # Return features and the dictionary of aggregated aux losses
        return x1_, x2_, x3_, x4_, x5_, total_aux_losses


class decoder(nn.Module):
    def __init__(self, d_text=512):
        super(decoder, self).__init__()
        self.moe_vit_block1 = HeterogeneousTransformerBlock(dim=256, d_text=d_text, num_experts=8, top_k=2)
        self.up1 = Upsample(n_feat=256)
        self.moe_vit_block2 = HeterogeneousTransformerBlock(dim=128, d_text=d_text, num_experts=8, top_k=2)
        self.up2 = Upsample(n_feat=128)
        self.moe_vit_block3 = HeterogeneousTransformerBlock(dim=64, d_text=d_text, num_experts=8, top_k=2)
        self.up3 = Upsample(n_feat=64)
        self.moe_vit_block4 = HeterogeneousTransformerBlock(dim=32, d_text=d_text, num_experts=8, top_k=2)
        self.up4 = Upsample(n_feat=32)

    def forward(self, x1, x2, x3, x4, x5, route_feature, task_id):
        # Initialize aggregated aux losses dictionary
        device = x5.device # Use device from an input tensor
        total_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                            'mi_loss': torch.tensor(0.0, device=device)}

        y5_moe, aux_losses1 = self.moe_vit_block1(x5, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses1.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses1.get('mi_loss', 0.0)
        y4_up = self.up1(y5_moe)
        y4 = y4_up + x4

        y4_moe, aux_losses2 = self.moe_vit_block2(y4, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses2.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses2.get('mi_loss', 0.0)
        y3_up = self.up2(y4_moe)
        y3 = y3_up + x3

        y3_moe, aux_losses3 = self.moe_vit_block3(y3, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses3.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses3.get('mi_loss', 0.0)
        y2_up = self.up3(y3_moe)
        y2 = y2_up + x2

        y2_moe, aux_losses4 = self.moe_vit_block4(y2, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses4.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses4.get('mi_loss', 0.0)
        y1_up = self.up4(y2_moe)
        y1 = y1_up + x1

        # Return final feature map and the dictionary of aggregated aux losses
        return y1, total_aux_losses




class Fusion(nn.Module):
    def __init__(self, n_class=9, d_text=512):
        super(Fusion, self).__init__()
        self.n_class = n_class
        self.encoder = encoder(d_text=d_text)
        self.decode = decoder(d_text=d_text)

        # Gated Fusion Blocks (remain the same)
        self.fusion_block1 = IndependentSpatialGatedFusionBlock(channels=16)
        self.fusion_block2 = IndependentSpatialGatedFusionBlock(channels=32)
        self.fusion_block3 = IndependentSpatialGatedFusionBlock(channels=64)
        self.fusion_block4 = IndependentSpatialGatedFusionBlock(channels=128)
        self.fusion_block5 = IndependentSpatialGatedFusionBlock(channels=256)
        # Task Head
        self.fusion_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.recon_head_vis = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.recon_head_inf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

        # DeFog, LowEnhance and Derain
        self.w1 = nn.Parameter(torch.tensor(0.1))
        self.w2 = nn.Parameter(torch.tensor(0.1))
        self.w3 = nn.Parameter(torch.tensor(0.1))



    def _reconstruct_and_segment(self, fused_image, route_feature):
        # Initialize aggregated aux losses dictionary for this path
        device = fused_image.device
        recon_seg_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                                'mi_loss': torch.tensor(0.0, device=device)}

        # 1. Encode the fused image (task_id=1 for auxiliary path gating)
        # Encoder output features: rf1, rf2, rf3, rf4, rf5
        # Encoder aux losses: aux_losses_enc (dictionary potentially containing 'standard_moe_loss', 'mi_loss')
        rf1, rf2, rf3, rf4, rf5, aux_losses_enc = self.encoder(fused_image, route_feature, task_id=1)

        # Aggregate losses from encoder
        recon_seg_aux_losses['standard_moe_loss'] += aux_losses_enc.get('standard_moe_loss', torch.tensor(0.0, device=device))
        recon_seg_aux_losses['mi_loss'] += aux_losses_enc.get('mi_loss', torch.tensor(0.0, device=device))

        # 2. Decode features ONCE (task_id=1 for auxiliary path gating)
        # Decoder output features: y1_decoded_features (e.g., shape [B, C, H, W])
        # Decoder aux losses: aux_losses_dec (dictionary potentially containing 'standard_moe_loss', 'mi_loss')
        y1_decoded_features, aux_losses_dec = self.decode(rf1, rf2, rf3, rf4, rf5, route_feature, task_id=1)

        # Aggregate losses from decoder
        recon_seg_aux_losses['standard_moe_loss'] += aux_losses_dec.get('standard_moe_loss', torch.tensor(0.0, device=device))
        recon_seg_aux_losses['mi_loss'] += aux_losses_dec.get('mi_loss', torch.tensor(0.0, device=device))

        # 3. Apply Reconstruction Heads to the single decoded feature map
        bw_vi = self.recon_head_vis(y1_decoded_features) # e.g., Output: [B, 3, H, W]
        bw_ir = self.recon_head_inf(y1_decoded_features) # e.g., Output: [B, 3, H, W]


        # Return the reconstruction/segmentation outputs and the aggregated aux losses dictionary
        return bw_vi, bw_ir, recon_seg_aux_losses

    def forward(self, vi, ir, route_feature):
        device = vi.device

        # Initialize aggregated aux losses dictionary for the main fusion path
        fusion_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                             'mi_loss': torch.tensor(0.0, device=device)}

        # --- 1. Encoding ---
        v1, v2, v3, v4, v5, aux_losses_enc_v = self.encoder(vi, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_enc_v.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_enc_v.get('mi_loss', 0.0)

        i1, i2, i3, i4, i5, aux_losses_enc_i = self.encoder(ir, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_enc_i.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_enc_i.get('mi_loss', 0.0)

        # --- 2. Feature Fusion ---
        f1 = self.fusion_block1(v1, i1)
        f2 = self.fusion_block2(v2, i2)
        f3 = self.fusion_block3(v3, i3)
        f4 = self.fusion_block4(v4, i4)
        f5 = self.fusion_block5(v5, i5)

        # --- 3. Fusion Feature Decoding ---
        y1_fused, aux_losses_dec_fus = self.decode(f1, f2, f3, f4, f5, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_dec_fus.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_dec_fus.get('mi_loss', 0.0)

        # --- 4. Generate Fusion Output ---
        fusion_res = self.fusion_head(y1_fused)

        # --- 5. Reconstruction & Segmentation (Training Only) ---
        if self.training:
            # Call the helper function which now returns aux losses for its path
            bw_vi, bw_ir, recon_seg_aux_losses = self._reconstruct_and_segment(
                fusion_res, # Keep attached, gradients can flow
                route_feature
            )

            # Combine aux losses from both paths
            total_aux_losses = {}
            total_aux_losses['standard_moe_loss'] = fusion_aux_losses.get('standard_moe_loss', 0.0) + \
                                                    recon_seg_aux_losses.get('standard_moe_loss', 0.0)
            total_aux_losses['mi_loss'] = fusion_aux_losses.get('mi_loss', 0.0) + \
                                          recon_seg_aux_losses.get('mi_loss', 0.0)

            # Return all outputs and the final combined aux losses dictionary
            #print(recon_seg_aux_losses.get('mi_loss', 0.0))
            return fusion_res, bw_vi, bw_ir, total_aux_losses
        else:
            # --- Evaluation Mode ---
            # Only return fusion result and the aux losses from the fusion path
            # Other outputs are None
            return fusion_res, None, None, fusion_aux_losses


