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
git clone --recursive https://github.com/matt576/image-editing.git
cd image-editing
git submodule update --init --recursive
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
pip install -e "./diffusers[torch]"
pip install diffusers["torch"] transformers
```
In case the torch version is not supporting CUDA, manually install the version.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 2. Functions demonstration:
### 1. Inpainting (Mask: SAM -> Diffusion: ControlNet)
#### Checkpoints:
###### SAM:
Model:  facebook/sam-vit-base <br />
Pipeline: facebook/sam-vit-base <br />
###### ControlNet:
Model: lllyasviel/control_v11p_sd15_inpaint <br />
Pipeline: runwayml/stable-diffusion-v1-5 <br />

Error: ControlNet generates black images for dtype.float16: <br />
Error: Output produced contains NSFW content -> set safety_checker=None, requires_safety_checker=False for loading pipe

Example: **python mask_func.py**

### 2. Conditional Inpainting (Mask: Manual -> Diffusion: ControlNet Inpainting)
#### Checkpoints:
###### ControlNet:
Model: lllyasviel/control_v11p_sd15_inpaint <br />
Pipeline: runwayml/stable-diffusion-v1-5 <br />

Example: **python inpaint_func.py**

### 3. Magic Eraser (Mask: SAM -> Diffusion: Latent Diffusion)
#### Checkpoints:
###### SAM:
Model:  facebook/sam-vit-base <br />
Pipeline: facebook/sam-vit-base <br />
###### Latent Diffusion:
Model: https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1

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

Example: **python inpaint_ldm.py --indir inputs/example_dog --outdir outputs/inpainting_results --steps 5**

### 3. Background Extraction (Mask: Depth Anything -> Diffusion: Latent Diffusion)
Additional imports:
```bash
pip install scikit-learn
```
The method applies the Depth-Anything model to perform depth estimation.
In order to extract the foreground and background of the image to apply background blurring/restyling we can use the following techniques:
1. Thresholding
2. K-Means
3. Fine-tune the model on foreground and background annotated using SAM
- Stanford Background Dataset: https://www.kaggle.com/datasets/balraj98/stanford-background-dataset (715 320x240 colored images)
- Fine-tuning necessary (!)
#### Checkpoints:
###### Depth Anything:
Model: LiheYoung/depth-anything-small-hf



