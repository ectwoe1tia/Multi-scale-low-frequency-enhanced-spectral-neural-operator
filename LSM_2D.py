"""
@author: Haixu Wu
"""
import torch.nn.functional as F
from pdebench.models.fno.fno import SpectralConv2d_fast
import torch.nn as nn
import torch
import numpy as np
import math


################################################################
# Multiscale modules 2D
################################################################
class F_SpectralConv2d(nn.Module):
    def __init__(self, modes_x=12, modes_y=12, in_dim=20, out_dim=20, forecast_ff=None, backcast_ff=None, fourier_weight=None,
                factor=1, num_block=1, mode='full'):
        super(F_SpectralConv2d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        mode_x:number of x flouier modes
        mode_y:number of y flouier modes
        in_dim:input channel
        out_dim:output channel

        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x 
        self.modes_y = modes_y
        self.mode = mode
        self.fourier_weight = fourier_weight
        self.num_block = num_block

        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y]:
                weight = torch.FloatTensor(num_block, in_dim//num_block, out_dim//num_block, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.backcast_ff = backcast_ff
        self.forecast_ff = forecast_ff

    def forward(self, x):

        x_temp = self.forward_fourier(x) + x

        b = self.backcast_ff(x_temp) 

        return b

    def forward_fourier(self, x):

        B, C, H, W = x.shape
        x = x.reshape(B, self.num_block, self.in_dim//self.num_block, H, W)
        #x = rearrange(x, 'b m n i -> b i m n')
        #x = x.permute(0,3,1,2)
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #dimension X,Y
        x_ft = torch.fft.rfft(x,dim=-2,norm='ortho')
        y_ft = torch.fft.rfft(x,dim=-1,norm='ortho')


        # Multiply relevant Fourier modes 
        # Store the Fourier transform calculation results in the x and y directions separately
        out_ft_x = torch.zeros(B, self.num_block, self.out_dim//self.num_block,
                               x.size(-2)//2+1, x.size(-1), device=x.device, dtype=torch.cfloat)
        out_ft_y = torch.zeros(B, self.num_block, self.out_dim//self.num_block,
                               x.size(-2), x.size(-1)//2+1, device=x.device, dtype=torch.cfloat)

        if self.mode == 'full':
            out_ft_y[:, :, :, :, :self.modes_y] = torch.einsum(
                "bkixy,kioy->bkoxy",
                y_ft[:, :, :, :, :self.modes_y],
                torch.view_as_complex(self.fourier_weight[1]))
            out_ft_x[:, :, :, :self.modes_x, :] = torch.einsum(
                "bkixy,kiox->bkoxy",
                x_ft[:, :, :, :self.modes_x, :],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft_x[:, :, :, :, :self.modes_y] = x_ft[:, :, :, :, :self.modes_y]
            out_ft_y[:, :, :, :self.modes_x, :] = y_ft[:, :, :, :self.modes_x, :]
        
        #Inverse Fourier transform
        out_ft_x=torch.fft.irfft(out_ft_x,H,dim=-2,norm='ortho')
        out_ft_y=torch.fft.irfft(out_ft_y,W,dim=-1,norm='ortho')
        x=out_ft_x+out_ft_y
        x = x.reshape(B, C, H, W)


        return x
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


################################################################
# Patchify and Neural Spectral Block
################################################################
class NeuralSpectralBlock2d(nn.Module):
    def __init__(self, width, num_basis, patch_size=[3, 3], num_token=4):
        super(NeuralSpectralBlock2d, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # basis
        self.modes_list = (1.0 / float(num_basis)) * torch.tensor([i for i in range(num_basis)],
                                                                  dtype=torch.float).cuda()
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float))
        # latent
        self.head = 8
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width)) * torch.rand(self.head, self.num_token, width // self.head, dtype=torch.float))
        self.encoder_attn = nn.Conv2d(self.width, self.width * 2, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv2d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, H, W = x.shape
        L = H * W
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        x_tmp = self.encoder_attn(x).view(B, C * 2, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head, 2).permute(4, 0, 2, 1, 3).contiguous()
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, H, W = x.shape
        L = H * W
        latent_token = latent_token.view(B, self.head, C // self.head, self.num_token).permute(0, 1, 3, 2).contiguous()
        x_tmp = self.decoder_attn(x).view(B, C, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head).permute(0, 2, 1, 3).contiguous()
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H, W) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        # patchify
        x = x.view(x.shape[0], x.shape[1],
                   x.shape[2] // self.patch_size[0], self.patch_size[0], x.shape[3] // self.patch_size[1],
                   self.patch_size[1]).contiguous() \
            .permute(0, 2, 4, 1, 3, 5).contiguous() \
            .view(x.shape[0] * (x.shape[2] // self.patch_size[0]) * (x.shape[3] // self.patch_size[1]), x.shape[1],
                  self.patch_size[0],
                  self.patch_size[1])
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = x.view(B, (H // self.patch_size[0]), (W // self.patch_size[1]), C, self.patch_size[0],
                   self.patch_size[1]).permute(0, 3, 1, 4, 2, 5).contiguous() \
            .view(B, C, H, W).contiguous()
        return x


class Model(nn.Module):
    def __init__(self, args, bilinear=True):
        super(Model, self).__init__()
        in_channels = args.in_dim
        out_channels = args.out_dim
        width = args.d_model
        num_token = args.num_token
        num_basis = args.num_basis
        patch_size = [int(x) for x in args.patch_size.split(',')]
        padding = [int(x) for x in args.padding.split(',')]
        # multiscale modules
        self.inc = DoubleConv(width, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(width * 8, width * 16 // factor)
        self.up1 = Up(width * 16, width * 8 // factor, bilinear)
        self.up2 = Up(width * 8, width * 4 // factor, bilinear)
        self.up3 = Up(width * 4, width * 2 // factor, bilinear)
        self.up4 = Up(width * 2, width, bilinear)
        self.outc = OutConv(width, width)
        # Patchified Neural Spectral Blocks
        self.process1 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)
        self.process2 = NeuralSpectralBlock2d(width * 2, num_basis, patch_size, num_token)
        self.process3 = NeuralSpectralBlock2d(width * 4, num_basis, patch_size, num_token)
        self.process4 = NeuralSpectralBlock2d(width * 8, num_basis, patch_size, num_token)
        self.process5 = NeuralSpectralBlock2d(width * 16 // factor, num_basis, patch_size, num_token)
        # self.w0 = nn.Conv2d(width, width, 1)
        # self.conv0 = SpectralConv2d_fast(width, width, 4, 4)
        # self.conv0 = F_SpectralConv2d(in_dim=width, out_dim=width, modes_x=4, modes_y=4, backcast_ff=F.gelu, forecast_ff=F.gelu)
        # projectors
        self.padding = padding
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x) 

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[1], :-self.padding[0]]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
