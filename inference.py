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
T = 1000
IMG_SIZE = 256
BATCH_SIZE = 1
learning_rate = 0.00001   # original is 0.001
epochs = 1000 # Try more!
# Image Folder
ImageF = 'target'
# Load model
load_dir = 'models/model_paper11.pt'   
load_dir2 = 'models/model2_paper11.pt'   
model = torch.load(load_dir)    
model2 = torch.load(load_dir2)  
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)   
model2.to(device)  

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

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    train_data=torchvision.datasets.ImageFolder(ImageF ,transform=data_transform)
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

@torch.no_grad()
def sample_timestep(x, x_0, y, t, model, model2):
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
    
    x_0 = x_0.to(device)
    #print("x - size: ", x.size())
    #print("x_0 - size: ", x_0.size())
    x_concatenated = torch.cat((x, x_0), dim=1)  # Concatenate error Expected size 1 but got size 5 for tensor number 1 in the list.
    y_concatenated = torch.cat((y, x_0), dim=1)
    #print("x_concatenated - size: ", x_concatenated.size())
    # Call model (current image - noise prediction)
    # Mean for X
    model_mean_X = sqrt_recip_alphas_t * (
        x - betas_t * model(x_concatenated, t) / sqrt_one_minus_alphas_cumprod_t
    )
    # Mean for Y 
    model_mean_Y = sqrt_recip_alphas_t * (
        y - betas_t * model(y_concatenated, t) / sqrt_one_minus_alphas_cumprod_t
    )    
    # B_t
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)   
    if t == T-1:       # t == T
        noise = torch.randn_like(x)
        x_t_minus_1 = model_mean_X + torch.sqrt(posterior_variance_t) * noise
        y_t_minus_1 = model2(x_t_minus_1, t)
        #print("t = T-1")
        return x_t_minus_1, y_t_minus_1
    elif t == 0:       # t == 1
        noise = torch.randn_like(x)
        x_t_minus_1 = model_mean_X # This is also a Fake Input
        y_t_minus_1 = model_mean_Y 
        #print("t = 0")
        return x_t_minus_1, y_t_minus_1
    else:            # t > 1  
        noise = torch.randn_like(x)
        x_t_minus_1 = model_mean_X + torch.sqrt(posterior_variance_t) * noise
        y_t_minus_1 = model_mean_Y + torch.sqrt(posterior_variance_t) * noise \
                    + model2(x_t_minus_1, t)
        y_t_minus_1 = (1/torch.sqrt(torch.tensor(2)))*(y_t_minus_1 - (2*sqrt_recip_alphas_t)*(y - betas_t * model(y_concatenated, t) / sqrt_one_minus_alphas_cumprod_t) ) \
                    + (1*sqrt_recip_alphas_t)*(y - betas_t * model(y_concatenated, t) / sqrt_one_minus_alphas_cumprod_t)
        #print("t > 1")
        return x_t_minus_1, y_t_minus_1

@torch.no_grad()
def sample_plot_image(epochs, T, IMG_SIZE, model, x_0, model2, save_fig_name):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    img2 = torch.randn((1, 3, img_size, img_size), device=device)   # It just a Fake Input
    plt.figure(figsize=(3,3))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img, img2 = sample_timestep(img, x_0, img2, t, model, model2)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        img2 = torch.clamp(img2, -1.0, 1.0)
    
    show_tensor_image(img2.detach().cpu())
    plt.show()
    plt.savefig(save_fig_name+"_y_0_paper11"+"IMG_SIZE_"+str(IMG_SIZE)+"_epochs_"+str(epochs)+"_T_"+str(T)+"_lr_"+str(learning_rate)+".png") 
    show_tensor_image(img.detach().cpu())
    plt.show()
    plt.savefig(save_fig_name+"_x_0_paper11"+"IMG_SIZE_"+str(IMG_SIZE)+"_epochs_"+str(epochs)+"_T_"+str(T)+"_lr_"+str(learning_rate)+".png") 
    plt.close()

if __name__ == '__main__':
    data=torchvision.datasets.ImageFolder(ImageF)
    show_images(data)
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    i = 0
    for step, batch in enumerate(dataloader):
        save_fig_name = "logs3/Inference_sample_plot_image_" + str(i)
        sample_plot_image(epochs, T, IMG_SIZE, model, batch[0], model2, save_fig_name)
        i=i+1
        
        
