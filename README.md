# AFM Image-Editing Tool

The AFM Image-Editing tool allows for AI-supported modification of images using a variety of technologies. <br />
Our repository includes implementation of the methods, examplary results and evaluation. <br />
Additionally, we created jupyter notebooks to better understand the possibilities of our Image-Editing tool and an app for running the methods on a webserver using Gradio. <br />

The tool supports the following functions:
- Mask generation
- Background blurring
- Background replacement
- Object removal
- Inpainting (Object Replacement)
- Outpainting
- Restyling
- Superresolution
- Txt2Img Image Generation

# 1. Set up the environment:
1. Clone the repo and initialize the submodules:

```bash
git clone --recursive https://github.com/matt576/image-editing.git
cd image-editing
git submodule update --init --recursive
```

2. To set up the working environment, conda is recommended.
Create and activate the new environment:

```bash
conda create -n pytorch_env python=3.8.10 \
  -c nvidia/label/cuda12.1.0 \
  -c conda-forge

conda activate image-editing-env
```
3. After creating and activating the environment, install the respective CUDA-supported Pytorch version:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
4. Install requirements for running diffusers and transformers:

```bash
pip install diffusers["torch"] transformers
cd diffusers
pip install -e "./diffusers[torch]"
cd ..
```
5. Additional requirements:

```bash
pip install omegaconf==2.1.1
pip install pytorch-lightning==1.6.1  # possibly newer version
pip install einops==0.3.0
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

pip install opencv-python

pip install -qr https://huggingface.co/briaai/RMBG-1.4/resolve/main/requirements.txt
pip install controlnet_aux
```

6. Additional setup for GroundedSAM submodule:

You should set the environment variable manually as described below, if you want to build a local GPU environment for Grounded-SAM:

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
Otherwise, the program will run on CPU-only mode. The following error will occur if you don't follow the previous steps: "NameError: name '_C' is not defined". <br />
If that happens, simply delete groundingDino from your environemnt and start over from setting the environemnt variable manually.

Install Segment Anything:

```bash
python -m pip install -e ../Grounded-Segment-Anything/segment_anything
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e ../Grounded-Segment-Anything/GroundingDINO # Follow previous CUDA_HOME steps carefully
```
In case you encounter path issues when running GroundedSAM-related functions, copy the GroundingDINO folder (found inside GroundedSAM submodule) into the code folder as an immediate temporary solution.

7. Install Gradio: <br />
Recommended version: 4.32.1
```bash
pip install gradio
```

8. LAMA setup:
```bash
pip install xxxx
```
*LIST NEXT REQUIREMENTS* LAMA
*MODELS TO DOWNLOAD*

# 2. Download the necessary model checkpoints for functions unavailable on huggingface diffusers/tranformers
Latent diffusion:
```bash
wget -O models/ldm_inpainting/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```
GroundedSAM (SAM + GroundingDINO):
```bash
wget -O models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

# 3. Modifications for compatibility purposes 
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

# 4. Functions demonstration:
In order to visualise the examples, we prepared a jupyter notebooks for each of the functions, located in the folder:

```bash
image-editing/code/notebooks/
```
Those serve as tutorials to help getting familiar with the variety of possibilities provided by the Image-Editing tool and generate outputs for examplary images.

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
/9-Txt2Img.ipynb
```

# 5. Gradio App - UI
In order to use all the capabilities of our tool in an user-friendly way, we developed an app in a form of webserver UI using the gradio library. <br />

This allows any user to upload their own images or artificially generate them via txt2img prompting, apply the aforementioned functionalities with different input parameters, visualize and download the results. <br />

Our program supports usage of more than just one model for a each function. The used model can be selected on the dropdown menus within each tab before generating the output image. <br />

In this section, we introduce the guide and steps of how to use the gradio app.

0. (Optional) Before launching the app, open the file
```bash
code/afm_gradio.py
```
and navigate to the main function to specify the input image- and mask paths you'd like to be opened when lauching the app. <br />
For that, simply edit the variables 'original_image_path' and 'input_mask_path' with your desired inputs. <br />
This is an optional step, since you can also upload an input image (e.g. via drag-and-drop) directly in the UI.

1. Launch the app:
```bash
cd code
python afm_gradio.py
```
After that, a public- and a local link for the accessing the UI will be generated. If you're using GPUs via remote access (SSH/Tunnel), we recommend using the public link.

2. Confirm your input image or open the accordion 'Txt2Img' if you'd like to generate your own input, and follow the instructions in the app.
3. Select your image editing task by clicking on a tab.
4. Within a tab, select the model on the dropdown menu and give the required inpit prompts by folllowing the instructions in the app.
5. Click 'Generate' and enjoy the app!

*Write some more sentences and steps, include screenshots / lists of methods etc*

# 6. Functions and used models
To allow users more flexibility, we included several models for each functionality. <br />
This section describes the implemented pipelines and exact model checkpoints used in the tool. <br />
Additionally this helps us evaluate the results and compare them to the state-of-the-art methods. <br />

## 1. Mask Generation
- SAM: facebook/sam-vit-huge
- Grounded-SAM: facebook/sam-vit-huge + GroundingDINO

## 2. Background-Blurring
- Depth-Anything pipeline
    - Depth estimation: LiheYoung/depth_anything_vitl14
    - Foreground extraction: briaai/RMBG-1.4


## 3. Background-Replacement
- Stable-diffusion pipelines: 
    - stabilityai/stable-diffusion-2-inpainting
    - diffusers/stable-diffusion-xl-1.0-inpainting-0.1 

## 4. Eraser
- Latent Diffusion pipelime: ldm_inpainting/last

- Lama pipeline: big-lama/best

## 5. Inpainting
- Stable-diffusion-v-1.5 pipeline: runwayml/stable-diffusion-inpainting
- Kandinsky-2.2 pipeline: kandinsky-community/kandinsky-2-2-decoder-inpaint
- Stable diffusion XL pipeline: diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- Stable diffusion with controlnet pipeline:
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

## 9. Txt2Img Image Generation
- Stable-diffusion-v-1.5: runwayml/stable-diffusion-v1-5
- Kandinsky-2.2: kandinsky-community/kandinsky-2-2-decoder
- Stable diffusion XL: stabilityai/stable-diffusion-xl-base-1.0

# 7. Evaluation
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
