import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from timeit import default_timer

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.complex64, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], torch.view_as_complex(self.weights1))
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], torch.view_as_complex(self.weights2))

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class F_SpectralConv2d(nn.Module):
    def __init__(self, modes_x=12, modes_y=12, in_dim=20, out_dim=20, forecast_ff=None, backcast_ff=None, fourier_weight=None,
                factor=1, num_block=1, mode='full'):
        super(F_SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
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

class Wcorrect2d(nn.Module):
    # W-cycle Module
    def __init__(self, width=64, latent_dim = 512, modes_x=16, modes_y=16, block_size=8, num_block=8, pool_mode = 'avg'):
        super(Wcorrect2d, self).__init__()
        self.width = width
        self.latent_dim = latent_dim
        self.num_block = num_block
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.pool_mode = pool_mode
        self.lim_mode = 8
        self.factor = 1

        self.DOWN = nn.AvgPool2d(2)

        self.conv0 = F_SpectralConv2d(in_dim=self.width, out_dim=self.width, num_block=self.num_block,
                                      modes_x=self.modes_x*self.factor, modes_y=self.modes_y*self.factor,
                                         forecast_ff=F.gelu,backcast_ff=F.gelu)
        self.conv1 = F_SpectralConv2d(in_dim=self.width, out_dim=self.width, num_block=self.num_block,
                                      modes_x=self.lim_mode, modes_y=self.lim_mode, 
                                      forecast_ff=F.gelu,backcast_ff=F.gelu)                

        self.UP = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):

        #32
        x = self.DOWN(x)
        x = self.conv0(x)

        #16
        x = self.DOWN(x)
        x = self.conv1(x)
        x = self.conv1(x)

        #32
        x = self.UP(x)
        x = self.conv0(x) 
        x = self.conv0(x) 

        #16
        x = self.DOWN(x)
        x = self.conv1(x) 
        x = self.conv1(x) 

        #32
        x = self.UP(x)
        x = self.conv0(x) 

        #64
        x = self.UP(x)

        return x

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        in_channels = args.in_dim
        out_channels = args.out_dim
        self.modes1 = args.num_basis
        self.modes2 = args.num_basis
        self.modes3 = args.num_basis // 2
        self.width = args.d_model
        self.layers = 4
        self.padding = [int(x) for x in args.padding.split(',')]

        self.spectral_layers = nn.ModuleList([])
        for _ in range(self.layers):
            self.spectral_layers.append(F_SpectralConv2d(in_dim=self.width,
                                                       out_dim=self.width,
                                                       modes_x=self.modes1,
                                                       modes_y=self.modes2,
                                                       forecast_ff=F.gelu,
                                                       backcast_ff=F.gelu,
                                                       ))

        self.correct0 = Wcorrect2d(width=self.width, modes_x=self.modes1, modes_y=self.modes2,pool_mode='avg')
        self.fc0 = nn.Linear(in_channels + 2, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        # Correct Module
        self.c0 = SpectralConv2d_fast(self.width, self.width, 4, 4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

        for i in range(self.layers):
            layer = self.spectral_layers[i]
            x = layer(x) 

        x = self.correct0(x) + x
        x = F.gelu(x)

        x = self.c0(x) + x

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