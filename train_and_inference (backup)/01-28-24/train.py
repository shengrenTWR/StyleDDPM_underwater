import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import math
from torch.optim import Adam
from utils import *
from utils_model import *


# Hyperparameters and parameter
T = 5000
IMG_SIZE = 128
BATCH_SIZE = 5
epochs = 1000 # Try more!
# Image Folder
ImageF = 'source'
ImageF2 = 'target'
# Save model's name
save_dir = 'models/model.pt' 
save_dir2 = 'models/model2.pt' 

# Define beta schedule
betas = linear_beta_schedule(timesteps=T)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def load_transformed_dataset(dirpath):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    train_data=torchvision.datasets.ImageFolder(dirpath ,transform=data_transform)
    return train_data
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

# ===================================Design loss function here =====================================
def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)    # 所以 u-net 就是想進辦法預測雜訊 （與輸入圖片等大小）
    return F.l1_loss(noise, noise_pred)
def get_loss2(model2, x_0, y_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    y_noisy, noise = forward_diffusion_sample(y_0, t, device)
    y_noisy_pred = model2(x_noisy, t)    # 所以 u-net 就是想進辦法預測雜訊 （與輸入圖片等大小）
    return F.l1_loss(y_noisy, y_noisy_pred)
# ==================================================================================================

@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image(epochs, T, IMG_SIZE, model, save_fig_name):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
    
    show_tensor_image(img.detach().cpu())
    plt.show()
    plt.savefig(save_fig_name+"IMG_SIZE_"+str(IMG_SIZE)+"_epochs_"+str(epochs)+"_T_"+str(T)+".png") 

if __name__ == '__main__':
    # Load image data
    data=torchvision.datasets.ImageFolder(ImageF)
    show_images(data)    
    data = load_transformed_dataset(ImageF)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    data2=torchvision.datasets.ImageFolder(ImageF2)
    show_images(data2)    
    data2 = load_transformed_dataset(ImageF2)
    dataloader2 = DataLoader(data2, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    
    # Simulate forward diffusion
    image = next(iter(dataloader))[0]
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        img, noise = forward_diffusion_sample(image, t)
        show_tensor_image(img)    
    
    # Define model
    model = SimpleUnet()
    model2 = SimpleUnet()
    # print("Num params: ", sum(p.numel() for p in model.parameters()))
    # print(model)    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model2.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    optimizer2 = Adam(model2.parameters(), lr=0.001)

    # Training (model1)
    for epoch in range(epochs):
        for step, (batch, batch2) in enumerate(zip(dataloader, dataloader2)):   # How to iterate over two dataloaders simultaneously?
          optimizer.zero_grad()
          optimizer2.zero_grad()

          t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
          loss = get_loss(model, batch[0], t)
          loss.backward()
          loss2 = get_loss2(model2, batch[0], batch2[0], t)
          loss2.backward()
          optimizer.step()
          optimizer2.step()

          if epoch % 5 == 0 and step == 0:
            print(f"Model1-> Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            print(f"Model2-> Epoch {epoch} | step {step:03d} Loss: {loss2.item()} ")
            save_fig_name = "logs/Model1_sample_plot_image_"
            save_fig_name2 = "logs/Model2_sample_plot_image_"
            sample_plot_image(epochs, T, IMG_SIZE, model, save_fig_name)
            sample_plot_image(epochs, T, IMG_SIZE, model2, save_fig_name2) # We should create a new one for transfer function
            
            
 
    # Save model
    torch.save(model, save_dir)
    torch.save(model2, save_dir2)