import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
import tqdm 
#!pip install denoising_diffusion_pytorch 
# %load_ext autoreload
# %autoreload 2


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def generate_x0(shape):
    weight = np.random.rand(10, 1)
    i = np.arange(shape)/15
    x_0 = 0
    for w in range(weight.shape[0]):    
        x_0 = x_0 + weight[w]*np.sin(w*i)
    return x_0/np.max(x_0)

x_0 = generate_x0(128)
print(x_0.shape)
plt.plot(x_0)
plt.show()
plt.savefig("squares.png") 



# Generate x_0 samples

num_x_0 = 1000
x_0_batch = []

for _ in range(num_x_0):
    x_0_batch.append(x_0 + np.random.rand(*x_0.shape)*0.1)

x_0_batch = np.stack(x_0_batch)
x_0_batch = x_0_batch/np.max(x_0_batch)
print(x_0_batch.shape)


model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=1
).to('cuda')

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 4000,   # number of steps
    # loss_type = 'l1'    # L1 or L2
).to('cuda')

x_0_batch_torch = torch.from_numpy(x_0_batch).reshape(num_x_0, 1, 128).type(torch.float).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, betas = (0.9, 0.99))

pbar = tqdm.tqdm(range(200))
for epoch in pbar:
    optimizer.zero_grad()
    loss = diffusion(x_0_batch_torch)
    loss.backward()
    # after a lot of training
    pbar.set_postfix({'Loss': loss.item()})
    optimizer.step()
    
sampled_images = diffusion.sample(batch_size = 4)

# plt.plot(x_0_batch[0,0,:])
plt.plot(sampled_images[0,0,:].detach().cpu().numpy())
plt.legend(['x_0','x_0 denoised'])
plt.savefig("squares2.png") 
