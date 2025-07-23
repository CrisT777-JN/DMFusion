import os

from PIL import Image
from torch.utils import data
from torchvision import transforms
import torch
import numpy as np
from models.common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])
SIZE = (400,400)

class MSRS_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor, phase=None):
        super().__init__()
        dirname = os.listdir(data_dir)
        self.phase = phase
        if self.phase == 'train':
            for sub_dir in dirname:
                temp_path = os.path.join(data_dir, sub_dir)
                if sub_dir == 'Inf':
                    self.inf_path = temp_path  
                if sub_dir == 'Vis':
                    self.vis_path = temp_path  
                if sub_dir == 'gt':
                    self.gt = temp_path
                elif sub_dir == 'seg_gt':
                    self.seg_gt_path = temp_path  
        elif self.phase == 'train1':
            for sub_dir in dirname:
                temp_path = os.path.join(data_dir, sub_dir)
                if sub_dir == 'Inf':
                    self.inf_path = temp_path  
                if sub_dir == 'Vis':
                    self.vis_path = temp_path  
                if sub_dir == 'gt':
                    self.gt = temp_path
                elif sub_dir == 'seg':
                    self.seg_gt_path = temp_path  

        else:
            for sub_dir in dirname:
                temp_path = os.path.join(data_dir, sub_dir)
                if sub_dir == 'Inf':
                    self.inf_path = temp_path  
                if sub_dir == 'Vis':
                    self.vis_path = temp_path 

        self.name_list = os.listdir(self.inf_path)  
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  
        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L') 
        vis_image = Image.open(os.path.join(self.vis_path, name))
        if self.phase =='train' or self.phase =='train1':
            inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L').resize(SIZE)  
            vis_image = Image.open(os.path.join(self.vis_path, name)).resize(SIZE)

            gt = Image.open(os.path.join(self.gt, name)).resize(SIZE)
            seg_gt = Image.open(os.path.join(self.seg_gt_path, name)).resize(SIZE)
        vis_image_clip = Image.open(os.path.join(self.vis_path, name)).resize((224,224))

        inf_image_clip = Image.open(os.path.join(self.inf_path, name)).resize((224,224))

        vis_image_clip = self.transform(vis_image_clip)
        inf_image_clip = self.transform(inf_image_clip)
        vis_image = self.transform(vis_image)
        c, _, _ = vis_image_clip.shape
        if c == 1:
            vis_image_clip = torch.cat([vis_image_clip]*3, dim=0)
        c, _, _ = vis_image.shape
        if c==1:
            vis_image = torch.cat([vis_image]*3, dim=0)
        vis_image_y, cr, cb = RGB2YCrCb(vis_image)
        if self.phase =='train' or self.phase == 'train1':
            gt = self.transform(gt)
            seg_gt = torch.tensor(np.array(seg_gt), dtype=torch.long)  
        inf_image = self.transform(inf_image)


        if self.phase == 'train' or self.phase == 'train1':
            return vis_image, inf_image, gt, seg_gt, vis_image_clip, name
        else:
            return vis_image_y, cr, cb, inf_image, vis_image_clip, inf_image_clip, name
    def __len__(self):
        return len(self.name_list)
