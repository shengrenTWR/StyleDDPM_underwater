import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import tqdm 

import numpy as np
import matplotlib.pyplot as plt

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 50,           # number of steps
)


training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
plt.plot(training_images[0,0,:,:])
# plt.legend(['x_0','x_0 denoised'])
plt.savefig("squares.png") 




# x_0_batch_torch = torch.from_numpy(x_0_batch).reshape(num_x_0, 1, 128).type(torch.float).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, betas = (0.9, 0.99))

pbar = tqdm.tqdm(range(50))
for epoch in pbar:
    optimizer.zero_grad()
    loss = diffusion(training_images)
    loss.backward()
    # after a lot of training
    pbar.set_postfix({'Loss': loss.item()})
    optimizer.step()
    
sampled_images = diffusion.sample(batch_size = 4)

# plt.plot(x_0_batch[0,0,:])
plt.plot(sampled_images[0,0,:,:])
# plt.legend(['x_0','x_0 denoised'])
plt.savefig("squares2.png") 