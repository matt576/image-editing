# Image-Editing

Image-Editing tool allows for AI-supported modification of images using a variety of technologies.
Currently, the following functions are supported:
- Background blurring
- Background replacement
- Object removal
- Inpainting / Outpaining
- Restyling
- Super-resolution

## 1. Set-up the environment:
In the first place clone the current project by using:

```bash
git clone https://github.com/matt576/image-editing.git
```

To set-up the working environment use conda. 
Create and activate the new environment.

```bash
conda create -n image-editing-env python=3.8.10
conda activate image-editing-env
```
To install the requirements in the environment to be able to run diffusers and transformers
run the following commands.
```bash
pip install -e "./diffusers[torch]"   # in diffusers 
pip install diffusers["torch"] transformers
# pip install xformers
```
In case the torch version is not supporting CUDA, manually install the version.

```bash
pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu118

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 2. Functions demonstration:
1. Inputation (Mask: SAM -> Diffusion (ControlNet) )
#### Checkpoints:
###### SAM:
Model:  facebook/sam-vit-base <br />
Pipeline: facebook/sam-vit-base <br />
###### ControlNet:
Model: lllyasviel/control_v11p_sd15_inpaint <br />
Pipeline: runwayml/stable-diffusion-v1-5 <br />

Error: ControlNet generates black images for dtype.float16: <br />
Error: Output produced contains NSFW content -> set safety_checker=None, requires_safety_checker=False for loading pipe