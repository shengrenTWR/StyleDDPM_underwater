# DDPM-based style transfer on underwater image 

This work mainly employed ddpm code from [colab](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing) and [UNET Implementation in PyTorch — Idiot Developer](https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201).

Some utilities from [denoising-diffusion-pytorch
](https://github.com/lucidrains/denoising-diffusion-pytorch) also included but isn't employed here.

## Setup

```
# Clone the repo.
git clone https://github.com/shengrenTWR/StyleDDPM_underwater.git
cd StyleDDPM_underwater

# Make a conda environment.
conda create --name StyleDDPM_underwater 
conda activate StyleDDPM_underwater

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

```


The work is inspired by


```bibtex
@article{lu2023underwater,
  title={Underwater image enhancement method based on denoising diffusion probabilistic model},
  author={Lu, Siqi and Guan, Fengxu and Zhang, Hanyu and Lai, Haitao},
  journal={Journal of Visual Communication and Image Representation},
  volume={96},
  pages={103926},
  year={2023},
  publisher={Elsevier}
}
```