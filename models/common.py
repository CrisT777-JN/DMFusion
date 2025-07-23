import torch
from torch import nn
import torch.nn.functional as F


class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out






def gradient(input):

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient



def clamp(value, min=0., max=1.0):

    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):


    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def RGB2YCrCb_Tensor(rgb_batch):

    Y_list = []
    Cb_list = []
    Cr_list = []
    for i in range(rgb_batch.shape[0]):
        Y, Cb, Cr = RGB2YCrCb(rgb_batch[i])
        Y_list.append(Y)
        Cb_list.append(Cb)
        Cr_list.append(Cr)
    Y = torch.stack(Y_list, dim=0)
    Cb = torch.stack(Cb_list, dim=0)
    Cr = torch.stack(Cr_list, dim=0)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):

    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out


def batch_YCrCb2RGB(Y_batch, Cb_batch, Cr_batch):
    
    # Concatenate Y, Cb, and Cr along the channel dimension -> (B, 3, H, W)
    ycrcb_batch = torch.cat([Y_batch, Cr_batch, Cb_batch], dim=1)

    # Flatten each image to (B, 3, H*W), then transpose -> (B, H*W, 3)
    im_flat = ycrcb_batch.view(ycrcb_batch.shape[0], 3, -1).transpose(1, 2)

    # Define the transformation matrix and bias
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).to(Y_batch.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y_batch.device)

    # Perform matrix operations to batch convert YCrCb to RGB
    temp = (im_flat + bias).matmul(mat)

    # Restore image shape (B, 3, H, W)
    out = temp.transpose(1, 2).view(Y_batch.shape[0], 3, Y_batch.shape[2], Y_batch.shape[3])

    # Clamp to the valid range
    out = clamp(out)

    return out




