import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageOps
from models.common import RGB2YCrCb


def pad_to_multiple_of_8(image, target_size):
    width, height = image.size
    pad_width = target_size[0] - width
    pad_height = target_size[1] - height
    padding = (0, 0, pad_width, pad_height)  
    return ImageOps.expand(image, padding, fill=0)

class MSRS_data(Dataset):
    def __init__(self,
                 root_dir,
                 size=None,
                 transform=None,
                 task=0,
                ):
        
        self.root_dir = root_dir
        self.transform = transform
        self.task = task
        if self.task ==0:
            gt_folder = os.path.join(root_dir, "Vis_gt")
        elif self.task ==1 or self.task==2:
            gt_folder = os.path.join(root_dir, "CT")

        self.gt_files = sorted([f for f in os.listdir(gt_folder) if os.path.isfile(os.path.join(gt_folder, f))])

        self.cliptransform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            
        ])
        self.size = size

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        
        filename = self.gt_files[idx]
        if self.task ==0:
            clean_vi_path = os.path.join(self.root_dir, "Vis_gt", filename)
            clean_ir_path = os.path.join(self.root_dir, "Inf_gt", filename)
            
            deg_ir_path = os.path.join(self.root_dir, "Inf", filename)
            deg_vi_path = os.path.join(self.root_dir, "Vis", filename)
        elif self.task == 1:
            clean_vi_path = os.path.join(self.root_dir, "CT_gt", filename)
            clean_ir_path = os.path.join(self.root_dir, "MRI_gt", filename)
            
            deg_vi_path = os.path.join(self.root_dir, "CT", filename)
            deg_ir_path = os.path.join(self.root_dir, "MRI", filename)
        elif self.task ==2:
            clean_vi_path = os.path.join(self.root_dir, "CT_gt", filename)
            clean_ir_path = os.path.join(self.root_dir, "MRI_gt", filename)
            
            deg_vi_path = os.path.join(self.root_dir, "CT_gt", filename)
            deg_ir_path = os.path.join(self.root_dir, "MRI_gt", filename)



        clean_vi_pil = Image.open(clean_vi_path)
        clean_ir_pil = Image.open(clean_ir_path).convert('L')
        deg_vi_pil = Image.open(deg_vi_path)
        deg_ir_pil = Image.open(deg_ir_path).convert('L')
        if self.size:
            clean_vi_pil = Image.open(clean_vi_path).resize(self.size)
            clean_ir_pil = Image.open(clean_ir_path).convert('L').resize(self.size)
            deg_vi_pil = Image.open(deg_vi_path).resize(self.size)
            deg_ir_pil = Image.open(deg_ir_path).convert('L').resize(self.size)
        
        target_width, target_height = clean_vi_pil.size
        
        clean_ir_pil = clean_ir_pil.resize((target_width, target_height), Image.BILINEAR)
        deg_vi_pil = deg_vi_pil.resize((target_width, target_height), Image.BILINEAR)
        deg_ir_pil = deg_ir_pil.resize((target_width, target_height), Image.BILINEAR)

       
        xx = 32
        if target_width % xx != 0 or target_height % xx != 0:
            padded_width = ((target_width + xx-1) // xx) * xx
            padded_height = ((target_height + xx-1) // xx) * xx

            clean_vi_pil = pad_to_multiple_of_8(clean_vi_pil, (padded_width, padded_height))
            clean_ir_pil = pad_to_multiple_of_8(clean_ir_pil, (padded_width, padded_height))
            deg_vi_pil = pad_to_multiple_of_8(deg_vi_pil, (padded_width, padded_height))
            deg_ir_pil = pad_to_multiple_of_8(deg_ir_pil, (padded_width, padded_height))
            
        input_vis_pil = deg_vi_pil
        input_ir_pil = deg_ir_pil



        if self.transform:

            input_vis = self.transform(input_vis_pil)
            input_ir = self.transform(input_ir_pil)
            clean_vis = self.transform(clean_vi_pil)
            clean_ir = self.transform(clean_ir_pil)
            c1, _, _ = input_vis.shape
            c2, _, _ = clean_vis.shape
            if c1!=1:
                input_vis, _, _ = RGB2YCrCb(input_vis)
            if c2!=1:
                clean_vis,_,_ = RGB2YCrCb(clean_vis)
        else:

            to_tensor = transforms.ToTensor()
            input_vis = to_tensor(input_vis_pil)
            input_ir = to_tensor(input_ir_pil)
            clean_vis = to_tensor(clean_vi_pil)
            clean_ir = to_tensor(clean_ir_pil)

            c1, _, _ = input_vis.shape
            c2, _, _ = clean_vis.shape
            if c1 != 1:
                input_vis, _, _ = RGB2YCrCb(input_vis)
            if c2 != 1:
                clean_vis, _, _ = RGB2YCrCb(clean_vis)
          


        clip_input_vi = self.cliptransform(input_vis_pil)
        clip_input_ir = self.cliptransform(input_ir_pil)

        return input_vis, input_ir, clean_vis, clean_ir, [clip_input_vi, clip_input_ir], filename

