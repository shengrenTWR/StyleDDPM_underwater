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

# Tasks
# 1. x_concatenated = torch.cat((x, x_0), dim=1) owing to the batch size. (solved)
# 2. Check x_0 if it needs to feed into sampling several time? 
# 180, 185, 133-135

## 1. Check training dataset if they are consistent.



# Hyperparameters and parameter
T = 1000
IMG_SIZE = 256
BATCH_SIZE = 1
learning_rate = 0.00001   # original is 0.001
epochs = 10 # Try more!
# Image Folder  Originaaly,   ImageF = 'source' ImageF2 = 'target'
ImageF2 = 'source'  
ImageF = 'target'
# Save model's name
save_dir = 'models/model_paper11.pt' 
save_dir2 = 'models/model2_paper11.pt' 

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
        #transforms.RandomHorizontalFlip(),
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
    x_0 = x_0.to(device)
    x_concatenated = torch.cat((x_noisy, x_0), dim=1)
    #print("x_concatenated - size: ", x_concatenated.size())
    noise_pred = model(x_concatenated, t)   
    return F.l1_loss(noise, noise_pred)
def get_loss2(model2, x_0, y_0, t):   # debugging here noise should be the same
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    y_noisy, noise = forward_diffusion_sample(y_0, t, device)
    y_noisy_pred = model2(x_noisy, t)  
    return F.l1_loss(y_noisy, y_noisy_pred)
# ==================================================================================================

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
    # Load image data
    data=torchvision.datasets.ImageFolder(ImageF)
    show_images(data)    
    data = load_transformed_dataset(ImageF)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE)
    
    data2=torchvision.datasets.ImageFolder(ImageF2)
    show_images(data2)    
    data2 = load_transformed_dataset(ImageF2)
    dataloader2 = DataLoader(data2, batch_size=BATCH_SIZE)
    
    
    # Simulate forward diffusion
    image = next(iter(dataloader))[0]
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        plt.gca().remove()
        img, noise = forward_diffusion_sample(image, t)
        show_tensor_image(img)    
    
    # Define model
    model = build_unet2()
    model2 = build_unet()
    # print("Num params: ", sum(p.numel() for p in model.parameters()))
    # print(model)    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model2.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer2 = Adam(model2.parameters(), lr=learning_rate)
    i = 1;
    # Training (model1)
    for epoch in range(epochs):
        for step, (batch, batch2) in enumerate(zip(dataloader, dataloader2)):   # How to iterate over two dataloaders simultaneously?
            optimizer.zero_grad()
            optimizer2.zero_grad()

            
            # Save dataset for debugging
#             data1, labels1 = batch
#             data2, labels2 = batch2
    
#             img1 = transforms.ToPILImage()(data1[0])
#             img2 = transforms.ToPILImage()(data2[0])
            
#             img1.save(f'logs_for_image/batch1_image_{i}.png')
#             img2.save(f'logs_for_image/batch2_image_{i}.png')
#             i = i + 1
            #=======================================================
            
            
            
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
            #print(batch[0].shape)
                save_fig_name = "logs3/Model1_sample_plot_image_"
                save_fig_name2 = "logs3/Model2_sample_plot_image_"
                sample_plot_image(epochs, T, IMG_SIZE, model, batch[0], model2, save_fig_name)
                sample_plot_image(epochs, T, IMG_SIZE, model, batch[0], model2, save_fig_name2) # We should create a new one for transfer function
            
            
 
    # Save model
    torch.save(model, save_dir)
    torch.save(model2, save_dir2)