import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.adam import Adam
from utils.params import get_args
from model_dict import get_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import math
import os

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()

TRAIN_PATH = os.path.join(args.data_path, './piececonst_r421_N1024_smooth1.mat')
TEST_PATH = os.path.join(args.data_path, './piececonst_r421_N1024_smooth2.mat')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'./data/output_{args.model}_Seed_{seed}_{timestamp}.txt'
f = open(file_name, 'a')

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

################################################################
# models
################################################################
model = get_model(args)
print(count_params(model))
def metric_fourier_func(pred, target, if_mean=True, Lx=1., Ly=1., Lz=1., iLow=4, iHigh=12):
    B,H,W = pred.shape
    pred = pred.reshape(B,1,H,W,1)
    target = target.reshape(B,1,H,W,1)
    # pred = pred.permute(0, 3, 1, 2)
    # target = target.permute(0, 3, 1, 2)
    

    pred, target = pred.cuda(), target.cuda()
    pred_F = torch.fft.fftn(pred, dim=[2, 3])
    target_F = torch.fft.fftn(target, dim=[2, 3])

    idxs = target.size()
    nb, nc, nt= idxs[0], idxs[1],idxs[-1]
    nx, ny = idxs[2:4]
    _err_F = torch.abs(pred_F - target_F) ** 2
    _err_test = torch.abs(target_F)**2
    err_F = torch.zeros([nb, nc, max(nx // 2, ny // 2), nt]).to(device)
    err_F_test = torch.zeros([nb, nc, max(nx // 2, ny // 2), nt]).to(device)
    for i in range(nx // 2):
        for j in range(ny // 2):
            # it = mt.floor(mt.sqrt(i ** 2 + j ** 2 + k ** 2))
            it = max(i,j)
            if it > max(nx // 2, ny // 2) - 1:
                continue
            err_F[:, :, it] += _err_F[:, :, i, j]
            err_F_test[:, :, it] += _err_test[:, :, i, j]
    _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny ) * Lx * Ly 
    _err_test = torch.sqrt(torch.mean(err_F_test, axis=0)) / (nx * ny ) * Lx * Ly 

    err_F_r = torch.zeros([nc, 3, nt]).to(device)
    err_F_a = torch.zeros([nc, 3, nt]).to(device)

    # relative fourier error
    err_F_r[:,0] += torch.mean(_err_F[:,:iLow], dim=1)/torch.mean(_err_test[:,:iLow], dim=1)  # low freq
    err_F_r[:,1] += torch.mean(_err_F[:,iLow:iHigh], dim=1)/torch.mean(_err_test[:,iLow:iHigh], dim=1)  # middle freq
    err_F_r[:,2] += torch.mean(_err_F[:,iHigh:], dim=1)/torch.mean(_err_test[:,iHigh:], dim=1)  # high freq
    
    # absolute fourier error
    err_F_a[:,0] += torch.mean(_err_F[:,:iLow], dim=1)  # low freq
    err_F_a[:,1] += torch.mean(_err_F[:,iLow:iHigh], dim=1)  # middle freq
    err_F_a[:,2] += torch.mean(_err_F[:,iHigh:], dim=1)  # high freq

    if if_mean:
        return torch.mean(err_F_a, dim=[0, -1]), torch.mean(err_F_r, dim=[0, -1])
    else:
        return err_F_a, err_F_r
################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = reader.read_field('sol')[:ntrain, ::r1, ::r2][:, :s1, :s2]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest, ::r1, ::r2][:, :s1, :s2]
y_test = reader.read_field('sol')[:ntest, ::r1, ::r2][:, :s1, :s2]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
y_normalizer.cuda()

x_train = x_train.reshape(ntrain, s1, s2, 1)
x_test = x_test.reshape(ntest, s1, s2, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False, d=2 ,p=2,Frequency=True)
testloss = LpLoss(size_average=False)



for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_ll1 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s1, s2)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))

        train_ll = testloss(out.view(batch_size, -1), y.view(batch_size, -1))
        train_ll1 += train_ll.item()

        loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        
    scheduler.step()

    model.eval()
    test_l2 = 0.0
    test_tl1 = 0.0
    with torch.no_grad():
        itot = 0
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s1, s2)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

            test_tl = testloss(out.view(batch_size, -1), y.view(batch_size, -1))
            test_tl1 += test_tl.item()

            _err_F_a, _err_F_r = metric_fourier_func(out, y)
            if itot == 0:
                err_F_a, err_F_r = _err_F_a, _err_F_r
            else:
                err_F_a += _err_F_a
                err_F_r += _err_F_r

                mean_dim = [i for i in range(len(y.shape)-2)]
                mean_dim.append(-1)
                mean_dim = tuple(mean_dim)
            itot += 1
        err_F_a = np.array(err_F_a.data.cpu()/itot)
        err_F_r = np.array(err_F_r.data.cpu()/itot)
        print('absolute MSE in Fourier space: {0}'.format(err_F_a))
        print('realtive MSE in Fourier space: {0}'.format(err_F_r))

        f.write('Epoch: {0}\n'.format(ep))
        f.write('absolute MSE in Fourier space: {0}\n'.format(err_F_a))
        f.write('realtive MSE in Fourier space: {0}\n'.format(err_F_r))


    train_l2 /= ntrain
    test_l2 /= ntest

    train_ll1 /= ntrain
    test_tl1 /= ntest

    t2 = default_timer()
    # print(ep, t2 - t1, train_l2, test_l2)
    print(ep, t2 - t1, train_ll1, test_tl1)
    f.write('{0} {1} {2} {3}\n'.format(ep, t2 - t1, train_ll1, test_tl1))

f.close()
# if ep % step_size == 0:
#     if not os.path.exists(model_save_path):
#         os.makedirs(model_save_path)
#     print('save model')
#     torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))
x_min, x_max, y_min, y_max = 0, 1, 0, 1
channel_plot = 0
model_name = model_save_name
target_plot = y[0]
pred_plot = out[0]
target_plot = target_plot.reshape(85,85,1,1)
pred_plot = pred_plot.reshape(85,85,1,1)
print(target_plot.size())
print(pred_plot.size())

# plot true value
fig, ax = plt.subplots(figsize=(6.75,6))
h = ax.imshow(pred_plot[...,-1,channel_plot].squeeze().t().detach().cpu(),
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
# h.set_clim(target_plot[...,-1,channel_plot].min(), target_plot[...,-1,channel_plot].max())
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)
# ax.set_title("Prediction", fontsize=15)
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)
ax.set_ylabel("$y$", fontsize=15)
ax.set_xlabel("$x$", fontsize=15)
plt.tight_layout()
filename = model_name + '_pred.png'
plt.savefig(filename, dpi=500)

fig, ax = plt.subplots(figsize=(6.75,6))
h = ax.imshow(target_plot[...,-1,channel_plot].squeeze().t().detach().cpu(),
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
# h.set_clim(target_plot[...,-1,channel_plot].min(), target_plot[...,-1,channel_plot].max())
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
filename = model_name + '_data.png'
plt.savefig(filename, dpi=500)

fig, ax = plt.subplots(figsize=(6.75,6))
diff = abs(target_plot-pred_plot)
h = ax.imshow((diff[...,-1,channel_plot]).squeeze().t().detach().cpu(),
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=0.0, vmax=0.00025)
# h.set_clim(diff[...,-1,channel_plot].min(), diff[...,-1,channel_plot].max())
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)
# ax.set_title("Data", fontsize=15)
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)
ax.set_ylabel("$y$", fontsize=15)
ax.set_xlabel("$x$", fontsize=15)
plt.tight_layout()
filename = model_name + '_diff.png'
plt.savefig(filename, dpi=500)

# plot fourier map
fig, ax = plt.subplots(figsize=(6.75,6))
target_F = torch.fft.fftn(target_plot, dim=[0, 1])
target_F = torch.fft.fftshift(target_F)
target_F = abs(target_F)
print(target_F.size())
target_F = torch.log(1+target_F)
h = ax.imshow((target_F[...,-1,channel_plot]).squeeze().t().detach().cpu(),
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
# h.set_clim(target_F[...,-1,channel_plot].min(), target_F[...,-1,channel_plot].max())
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
plt.savefig(filename, dpi=500)


fig, ax = plt.subplots(figsize=(6.75,6))
target_F = torch.fft.fftn(pred_plot, dim=[0, 1])
target_F = torch.fft.fftshift(target_F)
target_F = abs(target_F)
print(target_F.size())
target_F = torch.log(1+target_F)
h = ax.imshow((target_F[...,-1,channel_plot]).squeeze().t().detach().cpu(),
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
# h.set_clim(target_F[...,-1,channel_plot].min(), target_F[...,-1,channel_plot].max())
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
filename = model_name + '_pred_target_F.png'
plt.savefig(filename, dpi=500)

# plot diff map in fourier field
fig, ax = plt.subplots(figsize=(6.75,6))
target_F = torch.fft.fftn(target_plot, dim=[0, 1])
pred_F = torch.fft.fftn(pred_plot, dim=[0, 1])
diff = torch.abs(pred_F - target_F)
target_F = diff
target_F = torch.fft.fftshift(target_F)
target_F = abs(target_F)
target_F = torch.log(1+target_F)
h = ax.imshow((target_F[...,-1,channel_plot]).squeeze().t().detach().cpu(),
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=0.0, vmax=0.0175)
# h.set_clim(target_F[...,-1,channel_plot].min(), target_F[...,-1,channel_plot].max())
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)
# ax.set_title("Data", fontsize=30)
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)
ax.set_ylabel("$y$", fontsize=15)
ax.set_xlabel("$x$", fontsize=15)
plt.tight_layout()
filename = model_name + 'diff_target_F.png'
plt.savefig(filename, dpi=500)
