# Image-Editing

Image-Editing tool allows for AI-supported modification of images using a variety of technologies.
Our repository includes implementation of the methods, examplary results of the methods and evaluation.
Additionally, we created a jupyter notebook to better understand to possibilities of our Image-Editing tool and an app for running the methods on a webserver using gradio.
The tool supports the following functions:
- Mask generation
- Background blurring
- Background replacement
- Object removal
- Inpainting
- Outpainting
- Restyling
- Superresolution

# 1. Set-up the environment:
Clone the current project and initialize the submodules.

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

*LIST NEXT REQUIREMENTS*
*MODELS TO DOWNLOAD*

# 1.1. Download the necessary models unavailable on huggingface
Latent diffusion:
```bash
wget -O models/ldm_inpainting/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

Additional imports:
```bash
pip install omegaconf==2.1.1
pip install pytorch-lightning==1.6.1  # possibly newer version
pip install einops==0.3.0
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```
pip install opencv-python

pip install -qr https://huggingface.co/briaai/RMBG-1.4/resolve/main/requirements.txt
pip install controlnet_aux


# 1.2 Modifications for compatibility purposes 
Latent Diffusion:

In the file: **image-editing\src\taming-transformers\taming\data\utils.py**
change line 11:
```bash
from torch._six import string_classes
```
to
```bash
string_classes = str
```


# 2. Functions demonstration:
In order to visualise the examples, we prepared a jupyter notebooks for each of the functions located in

```bash
image-editing/code/notebooks/
```
Those generate outputs for examplary images and helps getting familiar with the variety of possibilities provided by the Image-Editing tool.

List of the available notebooks:
```bash
/1-Mask-Generation.ipynb
/2-Background-Blurring.ipynb
/3-Background-Replacement.ipynb
/4-Eraser.ipynb
/5-Inpainting.ipynb
/6-Outpainting.ipynb
/7-Restyling.ipynb
/8-Superresolution.ipynb
```

# 3. Gradio App - Demo
In order to use all the capabilities of our tool, we developed an app in a form of webserver usoing gradio.
This allows any user to generate images, apply the overmentioned functionalities, visualise and download the results.
Our program supports usage of more than just one model for a each function. The used model can be selected before generating the image.
In the following we introduce the guide and steps of how to use the gradio app.
*Write some more sentences and steps, include screenshots / lists of methods etc*
1.
2.
3.
4.
5.


# 4. Functions and used models
To allow users more flexibility, we included several models for each functionality.
This section describes the implemented pipelines and exact model checkpoints used in the tool.
Additionally this helps us evaluate the results and compare them to the state-of-the-art methods.

## 1. Mask Generation
- SAM: facebook/sam-vit-huge
- Grounded-SAM: 

## 2. Background-Blurring
- Depth-Anything pipeline
    - Depth estimation: LiheYoung/depth_anything_vitl14
    - Foreground extraction: briaai/RMBG-1.4


## 3. Background-Replacement
- Stable-diffusion pipeline: stabilityai/stable-diffusion-2-inpainting

## 4. Eraser
- Latent Diffusion pipelime: ldm_inpainting/last

- Lama pipeline: big-lama/best

## 5. Inpainting
- Stable-diffusion-v-1.5 pipeline: runwayml/stable-diffusion-inpainting
- Kandinsky-2.2 pipeline: kandinsky-community/kandinsky-2-2-decoder-inpaint
- Stable diffusion XL pipeline: diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- Stable diffusion with controlnet pipeline
    - Controlnet: lllyasviel/control_v11p_sd15_inpaint
    - Stable diffusion: runwayml/stable-diffusion-v1-5

## 6. Outpainting
- Stable diffusion pipeline: stabilityai/stable-diffusion-2-inpainting
- Stable diffusion XL pipeline: diffusers/stable-diffusion-xl-1.0-inpainting-0.1

## 7. Restyling:
- Stable diffusion pipeline: runwayml/stable-diffusion-v1-5
- Kandinsky pipeline: kandinsky-community/kandinsky-2-2-decoder
- Stable diffusion XL pipeline: stabilityai/stable-diffusion-xl-refiner-1.0

## 8. Superresolution:
- Latent diffusion pipeline: CompVis/ldm-super-resolution-4x-openimages
- Upscaler pipeline: stabilityai/stable-diffusion-x4-upscaler


# 5. Evaluation
## The evualuation results are stored in
```bash
image-editing/code/outputs/
```
and contain generated images for the introduced models.
The evaluation is basing on comaprison the models to state-of-the-art tools or end products distributed by companies like Google, Apple etc.
The overmentioned models are evaluated against:
1. Mask generation: 
2. Background blurring: Blur by Google Photos
3. Background replacement: SmartBackground by PicsArt
4. Object removal: Magic Eraser by Google Photos
5. Inpainting: DALL-E by OpenAI
6. Outpainting: DALL-E by OpenAI
7. Restyling: Image Style Transfer by PicsArt
8. Superresolution: Image Upscaler by PicsArt

---
---
---
---
---
---


### 5. GroundedSAM-based mask generation
You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:

First, to check which cuda versions are available and the required path:
```bash
module avail
```
Then:
```bash
source /etc/profile.d/lmod.sh  
module load cuda/12.1.0 # Should match cuda version from pytorch
echo $CUDA_HOME #check if variable was automatically set to /storage/software/cuda/cuda-12.1.0, otherwise set manually with EXPORT...
```
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/storage/software/cuda/cuda-12.1.0 # Path on atcremers60@in.tum.de
```

Install Segment Anything:

```bash
python -m pip install -e ../Grounded-Segment-Anything/segment_anything
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e ../Grounded-Segment-Anything/GroundingDINO # Follow previous CUDA_HOME steps carefully
```
#### Checkpoints:
```bash
wget -O models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
##### GroundingDINO:
code/models/groundingdino_swint_ogc.pth

Also, copy the GroundingDINO folder into the code folder as well as a temporary solution, since for some reason in the code folder without it, GroundingDino is not recognized as an import module inside the script.
##### SAM:
code/models/sam_vit_h_4b8939.pth

###### Input Command Exmple:
Specify via Text Prompt the object you want to detect and get the mask of. <br />
Until cudatoolkit and CUDA_PATH issues get resolved, the program runs on cpu only mode, so specify it in the respective flag. If device = "cuda", follwing error happnes if you dont follow the CUDA_HOME variable related steps in the grounding DINO installation:
"NameError: name '_C' is not defined"

```bash
python groundedsam_func.py   --config ../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py   --grounded_checkpoint models/groundingdino_swint_ogc.pth   --sam_checkpoint models/sam_vit_h_4b8939.pth   --input_image inputs/example_dog/dog.png   --output_dir "outputs/grounded_sam/"   --box_threshold 0.3   --text_threshold 0.25   --text_prompt "dog"   --device "cuda"
```
### 6. GroundedSAM-based inpainting
#### Checkpoints:
same as above
###### Input Command Exmple:
Specify via Text Prompt the object you want to detect and the object you want to replace it with. <br />

```bash
python groundedsam_inpaint.py   --config ../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py   --grounded_checkpoint models/groundingdino_swint_ogc.pth   --sam_checkpoint models/sam_vit_h_4b8939.pth   --input_image inputs/example_dog/dog.png   --output_dir "outputs/grounded_sam"   --box_threshold 0.3   --text_threshold 0.25   --det_prompt "dog"   --inpaint_prompt "bear cub, high quality, detailed"   --device "cuda"
 ```


### X. AFM Image Editing App

```bash
cd code
python afm_gradio.py
```
