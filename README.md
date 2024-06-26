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


Example:
```bash
python mask_func.py
```
### 2. Conditional Inpainting (Mask: Manual -> Diffusion: ControlNet Inpainting)
#### Checkpoints:
###### ControlNet:
Model: lllyasviel/control_v11p_sd15_inpaint <br />
Pipeline: runwayml/stable-diffusion-v1-5 <br />

Example:
```bash
python inpaint_func.py
```

### 3. Magic Eraser (Mask: SAM -> Diffusion: Latent Diffusion)
#### Checkpoints:
###### SAM:
Model:  facebook/sam-vit-base <br />
Pipeline: facebook/sam-vit-base <br />
###### Latent Diffusion:
Model: https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
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

Example:
```bash
python inpaint_ldm.py --indir inputs/example_dog --outdir outputs/inpainting_results --steps 5
```

### 4. Background Replacement (Portrait)

CUDA out of memory errors:
Models: 
- diffusers/controlnet-depth-sdxl-1.0
- OzzyGT/RealVisXL_V4.0_inpainting




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

### 7. Using the gradio app for groundedSAM

```bash
cd Grounded-Segment-Anything
python gradio_app.py
```

For it to run properly, the following modifications were performed:

Lines 196/197: change "image" to "composite" and "mask" to "layers" <br />
Line 372: input_image = gr.ImageEditor(sources='upload', type="pil", value="assets/demo2.jpg") <br />
Line 376: run_button = gr.Button() <br />
Lines 391-394: with gr.Column():
                gallery = gr.Gallery(
                label="Generated images", 
                show_label=False, 
                elem_id="gallery", 
                preview=True, 
                object_fit="scale-down"
                ) <br />
Line 399: just comment out or remove <br />
Line 400 (optional): change share=True if you need a public link <br />

### 8. Background Blurring (Depth Aynthing)
Additional imports:
```bash
pip install opencv-python
```
The method applies the Depth-Anything model to perform depth estimation.
In order to extract the foreground and background of the image to apply background blurring/restyling we can use the following techniques:
1. Splitting the output to several sections depending on the distance measurement
2. Thresholding and applying varable blurring filter
3. Merging the individual images to a single output

#### Checkpoints:
###### Depth Anything:
Model: LiheYoung/depth_anything_vitl14

Example:
```bash
python blurring.py
```
Errors: The path for directory needs to be modified. File Depth-Anything/depth_anything/dpt.py
in lines [147, 149] should have path leading to the model.

### 9. Background Replacement (RMBG-1.4)
Additional imports:
```bash
pip install -qr https://huggingface.co/briaai/RMBG-1.4/resolve/main/requirements.txt
pip install controlnet_aux
```
The method extracts the foreground subject from the original image.
In the following, 


#### Checkpoints:
###### Depth Anything:
Model: 

Errors: The path for directory needs to be modified. File Depth-Anything/depth_anything/dpt.py
in lines [147, 149] should have path leading to the model.

# Alternative models to compare results
1. Mask generation: RMBG-1.4 | GroundedSAM | SAM
2. Inpainting: ControlNet | ! Stable Diffusion ! | GroundedSAM
3. Outpainting: InpaintAnything (Replace)
4. Eraser: Latent Diffusion | ! LaMa !
5. Blurring: Depth Anything
6. Background Replacement (Portrait): ! Stable Diffusion ! |
7. Superresolution