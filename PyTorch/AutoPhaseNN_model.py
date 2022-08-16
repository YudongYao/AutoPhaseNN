import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from tqdm.notebook import tqdm
import numpy as np
from numpy.fft import fftn, fftshift


class Network(nn.Module):
    def __init__(self, argv):
        super(Network, self).__init__()
        self.argv = argv
        self.H, self.W = argv.shape, argv.shape
        self.T = argv.T
        self.nconv = argv.nconv
        self.use_down_stride = argv.use_down_stride
        self.use_up_stride = argv.use_up_stride
        self.n_blocks = argv.n_blocks

        def get_down_blocks():
            down_blocks_all = []
            n_filt_in = 1
            for block_indx in range(self.n_blocks):
                n_filt_out = self.nconv * 2 ** block_indx
                block = self.down_block(
                    n_filt_in, n_filt_out, self.use_down_stride)
                down_blocks_all += block
                n_filt_in = n_filt_out
            return down_blocks_all

        def get_up_blocks():
            n_filt_final = self.nconv * 2 ** (self.n_blocks-1)
            up_blocks_all = []
            n_filt_in = n_filt_final
            for block_indx in range(self.n_blocks-1, 0, -1):
                n_filt_out = self.nconv * 2 ** (block_indx - 1)
                block = self.up_block(
                    n_filt_in, n_filt_out, self.use_up_stride)
                up_blocks_all += block
                n_filt_in = n_filt_out
            up_blocks_all.append(
                nn.Conv3d(in_channels=self.nconv, out_channels=1, kernel_size=3,
                          stride=1, padding=(1, 1, 1))
            )
            return up_blocks_all

        self.encoder = nn.Sequential(*get_down_blocks())
        self.decoder1 = nn.Sequential(*get_up_blocks(), nn.Sigmoid())
        self.decoder2 = nn.Sequential(*get_up_blocks(), nn.Tanh())

    def down_block(self, filters_in, filters_out, use_down_stride=False):
        block = [
            nn.Conv3d(in_channels=filters_in, out_channels=filters_out,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(filters_out)]
        if use_down_stride:
            block += [
                nn.Conv3d(filters_out, filters_out, 3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out)]
        else:
            block += [
                nn.Conv3d(filters_out, filters_out, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out),
                nn.MaxPool3d((2, 2, 2))]
        return block

    def up_block(self, filters_in, filters_out, use_up_stride=False):
        block = [
            nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(filters_out)
        ]
        if use_up_stride:
            block += [
                nn.ConvTranspose3d(filters_out, filters_out,
                                   3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out)
            ]
        else:
            block += [
                nn.Conv3d(filters_out, filters_out, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out),
                nn.Upsample(scale_factor=2, mode='trilinear',
                            align_corners=True)
            ]

        return block

    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        # Normalize amp to max 1 before applying support
        amp = torch.clip(amp, min=0, max=1.0)

        mask = torch.tensor([0, 1], dtype=amp.dtype, device=amp.device)
        if self.argv.unsupervise:
            # Apply the support to amplitude
            amp = torch.where(amp < self.T, mask[0], amp)

        # Restore -pi to pi range
        # Using tanh activation (-1 to 1) for phase so multiply by pi
        ph = ph*np.pi

        # Pad the predictions to 2X
        pad = nn.ConstantPad3d(int(self.H/4), 0)
        amp = pad(amp)
        ph = pad(ph)

        # get support for viz
        support = torch.zeros(amp.shape, device=amp.device)
        support = torch.where(amp < self.T, mask[0], mask[1])

        # Create the complex number
        with torch.cuda.amp.autocast(enabled=False):
            complex_x = torch.complex(
                amp.float()*torch.cos(ph.float()), amp.float()*torch.sin(ph.float()))

        # Compute FT, shift and take abs
        y = torch.fft.fftn(complex_x, dim=(-3, -2, -1))
        # FFT shift will move the wrong dimensions if not specified
        y = torch.fft.fftshift(y, dim=(-3, -2, -1))
        y = torch.abs(y)

        # Normalize to scale_I
        if self.argv.scale_I > 0:
            max_I = torch.amax(y, dim=[-1, -2, -3], keepdim=True)
            y = self.argv.scale_I*torch.div(y, max_I+1e-6)  # Prevent zero div

        return y, complex_x, amp, ph, support
