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
In the first place clone the current project.

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
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 2. Functions demonstration:
### 1. Inpainting (Mask: SAM -> Diffusion (ControlNet) )
#### Checkpoints:
###### SAM:
Model:  facebook/sam-vit-base <br />
Pipeline: facebook/sam-vit-base <br />
###### ControlNet:
Model: lllyasviel/control_v11p_sd15_inpaint <br />
Pipeline: runwayml/stable-diffusion-v1-5 <br />

Error: ControlNet generates black images for dtype.float16: <br />
Error: Output produced contains NSFW content -> set safety_checker=None, requires_safety_checker=False for loading pipe

### 2. Object Removal (Mask: Manual -> Diffusion (ControlNet) )
#### Checkpoints:
###### Latent Diffusion:
Model: https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1 <br />
###### ControlNet:
Model: lllyasviel/control_v11p_sd15_inpaint <br />
Pipeline: runwayml/stable-diffusion-v1-5 <br />

Additional imports:
```bash
pip install omegaconf==2.1.1
pip install pytorch-lightning==1.6.1  # possibly newer version
pip install einops==0.3.0
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```
In the file: **image-editing\src\taming-transformers\taming\data\utils.py**

Change line 11:
```bash
from torch._six import string_classes
```
to
```bash
string_classes = str
```
Due to a no longer existing module _six since Pytorch 1.10. <br />
Error: ControlNet generates black images for dtype.float16: <br />
Error: Output produced contains NSFW content -> set safety_checker=None, requires_safety_checker=False for loading pipe

Example: ***python inpaint_ldm.py --indir inputs/example_dog --outdir outputs/inpainting_results --steps 5***
