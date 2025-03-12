# -*- coding: utf-8 -*-
import torch
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pdebench.models.fno.wfno import UnitGaussianNormalizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def metric_func(pred, target, if_mean=True, Lx=1., Ly=1., Lz=1., iLow=4, iHigh=12):
    """
    code for calculate metrics discussed in the Brain-storming session
    MSE, normalized MSE, max error, MSE at the boundaries, conserved variables, MSE in Fourier space, temporal sensitivity
    """
    pred, target = pred.to(device), target.to(device)
    # (batch, nx^i..., timesteps, nc)
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    # 50 1 2
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]

    # MSE (50,1,2)
    err_mean = torch.sqrt(torch.mean((pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2, dim=2))
    # (1,2)
    err_MSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nMSE = torch.mean(err_mean / nrm, dim=0)

    err_CSV = torch.sqrt(torch.mean(
        (torch.sum(pred.view([nb, nc, -1, nt]), dim=2) - torch.sum(target.view([nb, nc, -1, nt]), dim=2)) ** 2,
        dim=0))
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz
    # worst case in all the data
    err_Max = torch.max(torch.max(
        torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])), dim=2)[0], dim=0)[0]

    if len(idxs) == 4:  # 1D
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.), dim=0)
    elif len(idxs) == 5:  # 2D
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (2 * nx + 2 * ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    elif len(idxs) == 6:  # 3D
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0] - target[:, :, :, :, 0]) ** 2
        err_BD_z += (pred[:, :, :, :, -1] - target[:, :, :, :, -1]) ** 2
        err_BD = torch.sum(err_BD_x.reshape([nb, -1, nt]), dim=-2) \
                 + torch.sum(err_BD_y.reshape([nb, -1, nt]), dim=-2) \
                 + torch.sum(err_BD_z.reshape([nb, -1, nt]), dim=-2)
        err_BD = err_BD / (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
        err_BD = err_BD.reshape(1,1)

    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = torch.sqrt(torch.mean(torch.abs(pred_F - target_F) ** 2, axis=0)) / nx * Lx
    if len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        _err_test = torch.abs(target_F)**2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(device)
        err_F_test = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i ** 2 + j ** 2))
                if it > min(nx // 2, ny // 2) - 1:
                    continue
                err_F[:, :, it] += _err_F[:, :, i, j]
                err_F_test[:, :, it] += _err_test[:, :, i, j]
        eps=1e-6
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny) * Lx * Ly
        _err_test = torch.sqrt(torch.mean(err_F_test, axis=0)) / (nx * ny) * Lx * Ly

    elif len(idxs) == 6:  # 3D

        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        _err_F = torch.abs(pred_F - target_F) ** 2
        _err_test = torch.abs(target_F)**2

        err_F = torch.zeros([nb, nc, max(nx // 2, ny // 2, nz // 2), nt]).to(device)
        err_F_test = torch.zeros([nb, nc, max(nx // 2, ny // 2, nz // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                for k in range(nz // 2):
                    # it = mt.floor(mt.sqrt(i ** 2 + j ** 2 + k ** 2))
                    it = max(i,j,k)
                    if it > max(nx // 2, ny // 2, nz // 2) - 1:
                        continue
                    err_F[:, :, it] += _err_F[:, :, i, j, k]
                    err_F_test[:, :, it] += _err_test[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny * nz) * Lx * Ly * Lz
        _err_test = torch.sqrt(torch.mean(err_F_test, axis=0)) / (nx * ny * nz) * Lx * Ly * Lz

    err_F = torch.zeros([nc, 3, nt]).to(device)
    err_F_test = torch.zeros([nc, 3, nt]).to(device)
    err_F_test[:,0] += torch.mean(_err_test[:,:iLow], dim=1)  # low freq
    err_F_test[:,1] += torch.mean(_err_test[:,iLow:iHigh], dim=1)  # middle freq
    err_F_test[:,2] += torch.mean(_err_test[:,iHigh:], dim=1)  # high freq
    # print(err_F_test)
    
    # absolute fRMSE
    err_F[:,0] += torch.mean(_err_F[:,:iLow], dim=1)  # low freq
    err_F[:,1] += torch.mean(_err_F[:,iLow:iHigh], dim=1)  # middle freq
    err_F[:,2] += torch.mean(_err_F[:,iHigh:], dim=1)  # high freq
    
    # relative fRMSE
    # err_F[:,0] += torch.mean(_err_F[:,:iLow], dim=1)/torch.mean(_err_test[:,:iLow], dim=1)  # low freq
    # err_F[:,1] += torch.mean(_err_F[:,iLow:iHigh], dim=1)/torch.mean(_err_test[:,iLow:iHigh], dim=1)  # middle freq
    # err_F[:,2] += torch.mean(_err_F[:,iHigh:], dim=1)/torch.mean(_err_test[:,iHigh:], dim=1)  # high freq

    if if_mean:
        return torch.mean(err_MSE, dim=[0, -1]), \
               torch.mean(err_nMSE, dim=[0, -1]), \
               torch.mean(err_CSV, dim=[0, -1]), \
               torch.mean(err_Max, dim=[0, -1]), \
               torch.mean(err_BD, dim=[0, -1]), \
               torch.mean(err_F, dim=[0, -1])
    else:
        return err_MSE, err_nMSE, err_CSV, err_Max, err_BD, err_F

def metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot, model_name, x_min,
            x_max, y_min, y_max, t_min, t_max, mode='FNO', initial_step=None, ):
    if mode=='Unet':
        with torch.no_grad():
            itot = 0
            for xx, yy in val_loader:
                xx = xx.to(device)
                yy = yy.to(device)

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                    inp = inp.permute(temp_shape)
                    
                    y = yy[..., t:t+1, :]
                
                    temp_shape = [0]
                    temp_shape.extend([i for i in range(2,len(inp.shape))])
                    temp_shape.append(1)
                    im = model(inp).permute(temp_shape).unsqueeze(-2)
                    pred = torch.cat((pred, im), -2)
                    print(xx[..., 1:, :].size())
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                _err_MSE, _err_nMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                    = metric_func(pred, yy, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

                if itot == 0:
                    err_MSE, err_nMSE, err_CSV, err_Max, err_BD, err_F \
                        = _err_MSE, _err_nMSE, _err_CSV, _err_Max, _err_BD, _err_F
                    pred_plot = pred[:1]
                    target_plot = yy[:1]
                    val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                else:
                    err_MSE += _err_MSE
                    err_nMSE += _err_nMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F
                    
                    mean_dim = [i for i in range(len(yy.shape)-2)]
                    mean_dim.append(-1)
                    mean_dim = tuple(mean_dim)
                    val_l2_time += torch.sqrt(torch.mean((pred-yy)**2, dim=mean_dim))
                
                itot += 1

    elif mode=='FNO':
        with torch.no_grad():
            itot = 0
            for xx, yy, grid in val_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                grid = grid.to(device)

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    y = yy[..., t:t + 1, :]
                    im = model(inp, grid)

                    if t==initial_step:
                        pred=im
                    else:
                        pred = torch.cat((pred, im), -2)

                    #FNO:y,metric y
                    xx = torch.cat((xx[..., 1:, :], y), dim=-2)

                _err_MSE, _err_nMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                    = metric_func(pred, yy[..., initial_step:, :], if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)
                yy = yy[..., initial_step:, :]
                if itot == 0:
                    err_MSE, err_nMSE, err_CSV, err_Max, err_BD, err_F \
                        = _err_MSE, _err_nMSE, _err_CSV, _err_Max, _err_BD, _err_F
                    pred_plot = pred[:1]
                    target_plot = yy[:1]
                    val_l2_time = torch.zeros(pred.shape[-2]).to(device)
                else:
                    err_MSE += _err_MSE
                    err_nMSE += _err_nMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F
                    
                    mean_dim = [i for i in range(len(yy.shape)-2)]
                    mean_dim.append(-1)
                    mean_dim = tuple(mean_dim)
                    val_l2_time += torch.sqrt(torch.mean((pred-yy)**2, dim=mean_dim))

                itot += 1

    elif mode == "PINN":
        raise NotImplementedError#抛出异常


    err_MSE = np.array(err_MSE.data.cpu()/itot)
    err_nMSE = np.array(err_nMSE.data.cpu()/itot)
    err_CSV = np.array(err_CSV.data.cpu()/itot)
    err_Max = np.array(err_Max.data.cpu()/itot)
    err_BD = np.array(err_BD.data.cpu()/itot)
    err_F = np.array(err_F.data.cpu()/itot)
    print('MSE: {0:.5f}'.format(err_MSE))
    print('normalized MSE: {0:.5f}'.format(err_nMSE))
    print('MSE of conserved variables: {0:.5f}'.format(err_CSV))
    print('Maximum value of rms error: {0:.5f}'.format(err_Max))
    print('MSE at boundaries: {0:.5f}'.format(err_BD))
    print('MSE in Fourier space: {0}'.format(err_F))
    
    val_l2_time = val_l2_time/itot

    if plot:
        dim = len(yy.shape) - 3
        plt.ioff()
        if dim == 1:
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(pred_plot[...,channel_plot].squeeze().detach().cpu(),
                       extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,channel_plot].min(), target_plot[...,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Prediction", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$x$", fontsize=30)
            ax.set_xlabel("$t$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_pred.pdf'
            plt.savefig(filename)
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(target_plot[...,channel_plot].squeeze().detach().cpu(),
                       extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,channel_plot].min(), target_plot[...,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$x$", fontsize=30)
            ax.set_xlabel("$t$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_data.pdf'
            plt.savefig(filename)
    
        elif dim == 2:
            
            # pred data
            fig, ax = plt.subplots(figsize=(6.75,5))
            h = ax.imshow(pred_plot[...,-1,channel_plot].squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,-1,channel_plot].min(), target_plot[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=20)
            ax.set_title("Prediction", fontsize=20)
            ax.tick_params(axis='x',labelsize=20)
            ax.tick_params(axis='y',labelsize=20)
            ax.set_ylabel("$y$", fontsize=20)
            ax.set_xlabel("$x$", fontsize=20)
            plt.tight_layout()
            filename = model_name + '_pred.pdf'
            plt.savefig(filename)
            
            # target data
            fig, ax = plt.subplots(figsize=(6.75,5))
            h = ax.imshow(target_plot[...,-1,channel_plot].squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,-1,channel_plot].min(), target_plot[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=15)
            ax.set_title("Data", fontsize=10)
            ax.tick_params(axis='x',labelsize=15)
            ax.tick_params(axis='y',labelsize=15)
            ax.set_ylabel("$y$", fontsize=15)
            ax.set_xlabel("$x$", fontsize=15)
            plt.tight_layout()
            filename = model_name + '_data.png'
            plt.savefig(filename)

            # difference
            fig, ax = plt.subplots(figsize=(6.75,6))
            diff = abs(target_plot-pred_plot)
            h = ax.imshow((diff[...,-1,channel_plot]).squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(diff[...,-1,channel_plot].min(), diff[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=20)
            ax.set_title("Data", fontsize=20)
            ax.tick_params(axis='x',labelsize=20)
            ax.tick_params(axis='y',labelsize=20)
            ax.set_ylabel("$y$", fontsize=20)
            ax.set_xlabel("$x$", fontsize=20)
            plt.tight_layout()
            filename = model_name + '_diff.pdf'
            plt.savefig(filename)

            # plot fourier map
            fig, ax = plt.subplots(figsize=(6.5,6))
            target_F = torch.fft.fftn(target_plot, dim=[1, 2])
            target_F = torch.fft.fftshift(target_F)
            target_F = abs(target_F)
            target_F = torch.log(1+target_F)
            h = ax.imshow((target_F[...,-1,channel_plot]).squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_F[...,-1,channel_plot].min(), target_F[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=15)
            # ax.set_title("Data", fontsize=10)
            ax.tick_params(axis='x',labelsize=15)
            ax.tick_params(axis='y',labelsize=15)
            ax.set_ylabel("$y$", fontsize=15)
            ax.set_xlabel("$x$", fontsize=15)
            plt.tight_layout()
            filename = model_name + '_target_F.png'
            plt.savefig(filename)

            # plot difference map in fourier field
            fig, ax = plt.subplots(figsize=(6.75,5))
            target_F = torch.fft.fftn(target_plot, dim=[1, 2])
            pred_F = torch.fft.fftn(pred_plot, dim=[1, 2])
            diff = torch.abs(pred_F - target_F)
            target_F = diff
            target_F = torch.fft.fftshift(target_F)
            target_F = abs(target_F)
            target_F = torch.log(1+target_F)
            h = ax.imshow((target_F[...,-1,channel_plot]).squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_F[...,-1,channel_plot].min(), target_F[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            filename = model_name + 'diff_target_F.pdf'
            plt.savefig(filename)
        

    return err_MSE, err_nMSE, err_CSV, err_Max, err_BD, err_F


# LpLoss Function  
class LpLoss(object):
    """
    Lp loss function 
    """
    def __init__(self, p=2, reduction='mean'):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert p > 0
        self.p = p
        self.reduction = reduction
    def __call__(self, x, y, eps=1e-20):
        num_examples = x.size()[0]
        _diff = x.view(num_examples,-1) - y.view(num_examples,-1)
        _diff = torch.norm(_diff, self.p, 1)
        _norm = eps + torch.norm(y.view(num_examples,-1), self.p, 1)
        if self.reduction in ['mean']:
            return torch.mean(_diff/_norm)
        if self.reduction in ['sum']:
            return torch.sum(_diff/_norm)
        return _diff/_norm

# FftLoss Function  
class FftLpLoss(object):

    def __init__(self, p=2, reduction='mean'):
        super(FftLpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert p > 0
        self.p = p
        self.reduction = reduction
    def __call__(self, x, y, flow=None,fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:]
        dims = list(range(1,len(x.shape)))
        xf = torch.fft.fftn(x,dim=dims)
        yf = torch.fft.fftn(y,dim=dims)
        if flow is None: flow = 0
        if fhigh is None: fhigh = np.max(xf.shape[1:])

        if len(others_dims) ==1:
            xf = xf[:,flow:fhigh]
            yf = yf[:,flow:fhigh]        
        if len(others_dims) ==2:
            xf = xf[:,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh]
        if len(others_dims) ==3:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh]
        if len(others_dims) ==4:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]

        _diff = xf - yf.reshape(xf.shape)
        _diff = torch.norm(_diff.reshape(num_examples,-1), self.p, 1)
        _norm = eps + torch.norm(yf.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction in ['mean']:
            return torch.mean(_diff/_norm)
        if self.reduction in ['sum']:
            return torch.sum(_diff/_norm)
        return _diff/_norm

import torch.nn.functional as F
# FftLoss Function  
class FftMseLoss(object):

    def __init__(self, reduction='mean'):
        super(FftMseLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        self.reduction = reduction
    def __call__(self, x, y, flow=None,fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:-2]
        for d in others_dims:
            assert (d>1), "we expect the dimension to be the same and greater the 1"
        # print(others_dims)
        dims = list(range(1,len(x.shape)-1))
        xf = torch.fft.fftn(x,dim=dims)
        yf = torch.fft.fftn(y,dim=dims)
        if flow is None: flow = 0
        if fhigh is None: fhigh = np.max(xf.shape[1:])

        if len(others_dims) ==1:
            xf = xf[:,flow:fhigh]
            yf = yf[:,flow:fhigh]        
        if len(others_dims) ==2:
            xf = xf[:,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh]
        if len(others_dims) ==3:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh]
        if len(others_dims) ==4:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
        _diff = xf - yf
        _diff = _diff.reshape(num_examples,-1).abs()**2
        if self.reduction in ['mean']:
            return torch.mean(_diff).abs()
        if self.reduction in ['sum']:
            return torch.sum(_diff).abs()
        return _diff.abs()

