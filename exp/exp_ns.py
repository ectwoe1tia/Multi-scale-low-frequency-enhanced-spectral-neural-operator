import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.params import get_args
from model_dict import get_model
from utils.adam import Adam
import math
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
seed=3407
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
################################################################
# configs
################################################################
args = get_args()

TRAIN_PATH = os.path.join(args.data_path, './NavierStokes_V1e-5_N1200_T20.mat')
TEST_PATH = os.path.join(args.data_path, './NavierStokes_V1e-5_N1200_T20.mat')

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
T_in = args.T_in
T_out = args.T_out

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'./data/output_ns_{args.model}_Seed_{seed}_{timestamp}.txt'
f = open(file_name, 'a')

################################################################
# models
################################################################
model = get_model(args)
print(count_params(model))

################################################################
# load data and data normalization
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain, ::r1, ::r2, :T_in]
train_u = reader.read_field('u')[:ntrain, ::r1, ::r2, T_in:T_in + T_out]

test_a = reader.read_field('u')[-ntest:, ::r1, ::r2, :T_in]
test_u = reader.read_field('u')[-ntest:, ::r1, ::r2, T_in:T_in + T_out]

print(train_u.shape)
print(test_u.shape)

train_a = train_a.reshape(ntrain, s1, s2, T_in)
test_a = test_a.reshape(ntest, s1, s2, T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

textloss = LpLoss(d=1,p=1,size_average=False)
myloss = LpLoss(size_average=False)
step = 1
epochs=1

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += textloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = textloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)
            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain,
          test_l2_step / ntest / (T_out / step),
          test_l2_full / ntest)
    f.write('Epoch: {0}, train_l2_step: {1}, train_l2_full: {2}, test_l2_step: {3}, test_l2_full: {4}\n'.format(ep, train_l2_step / ntrain / (T_out / step), 
            train_l2_full / ntrain, test_l2_step / ntest / (T_out / step), test_l2_full / ntest))
    if ep % step_size == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))

for xx, yy in test_loader:
    loss = 0
    t=9
    xx = xx.to(device)
    yy = yy.to(device)
    y = yy[..., t:t + 10]
    break
model_save_path = args.model_save_path
model_save_name = args.model_save_name
x_min, x_max, y_min, y_max = 0, 1, 0, 1
channel_plot = 0
model_name = model_save_name

print(yy.size())
# print(yy[:,:,:,-1].size())
# print(yy[0,:,:,-1].size())
# print(yy[0,:,:,-1].reshape(64,64,1,1)[...,-1,channel_plot].size())
target_plot = yy
pred_plot = pred[0, :, :, :]  # 取出第一个样本的所有时间帧
model_name = model_save_name


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
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=0.0, vmax=2.00)
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
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=0.0, vmax=5.00)
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
            extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
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
print(filename)
plt.savefig(filename, dpi=500)