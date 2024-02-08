import torch
import matplotlib.pyplot as plt
from torch import nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

    
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs, t,):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.bn1(x)
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel     Add time embedding here 
        x = x + time_emb
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        return x
    
    
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        time_emb_dim = 32
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )        
        self.conv = conv_block(in_c, out_c, time_emb_dim)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        x = self.conv(inputs, t)
        p = self.pool(x)
        return x, p    
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        time_emb_dim = 32
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )        
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, time_emb_dim)
    def forward(self, inputs, skip, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, t)
        return x
    
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        time_emb_dim = 32
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )  
        # """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        # """ Bottleneck """
        self.b = conv_block(512, 1024, time_emb_dim)
        # """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        # """ Classifier """
        self.outputs = nn.Conv2d(64, 3, kernel_size=1, padding=0)
    def forward(self, inputs, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        #""" Encoder """
        s1, p1 = self.e1(inputs, timestep)
        s2, p2 = self.e2(p1, timestep)
        s3, p3 = self.e3(p2, timestep)
        s4, p4 = self.e4(p3, timestep)
        # """ Bottleneck """
        b = self.b(p4, t)
        # """ Decoder """
        d1 = self.d1(b, s4, timestep)
        d2 = self.d2(d1, s3, timestep)
        d3 = self.d3(d2, s2, timestep)
        d4 = self.d4(d3, s1, timestep)
        # """ Classifier """
        outputs = self.outputs(d4)
        return outputs 
    
    
    
    
    
    
class build_unet2(nn.Module):
    def __init__(self):
        super().__init__()
        time_emb_dim = 32
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )  
        # """ Encoder """
        self.e1 = encoder_block(6, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        # """ Bottleneck """
        self.b = conv_block(512, 1024, time_emb_dim)
        # """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        # """ Classifier """
        self.outputs = nn.Conv2d(64, 3, kernel_size=1, padding=0)
    def forward(self, inputs, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        #""" Encoder """
        s1, p1 = self.e1(inputs, timestep)
        s2, p2 = self.e2(p1, timestep)
        s3, p3 = self.e3(p2, timestep)
        s4, p4 = self.e4(p3, timestep)
        # """ Bottleneck """
        b = self.b(p4, t)
        # """ Decoder """
        d1 = self.d1(b, s4, timestep)
        d2 = self.d2(d1, s3, timestep)
        d3 = self.d3(d2, s2, timestep)
        d4 = self.d4(d3, s1, timestep)
        # """ Classifier """
        outputs = self.outputs(d4)
        return outputs 
    
    
    
    
    
    
    
