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

### 3. GroundedSAM-based mask generation
```bash
git submodule add https://github.com/IDEA-Research/Grounded-Segment-Anything.git
git submodule update --init --recursive
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
#### Checkpoints:
##### GroundingDINO:
code/models/groundingdino_swint_ogc.pth

Also, copy the GroundingDINO folder into the code folder as well as a temporary solution, since for some reason in the code folder without it, GroundingDino is not recognized as an import module inside the script.
##### SAM:
code/models/sam_vit_h_4b8939.pth

###### Input Command Exmple:
Specify via Text Prompt the object you want to detect and get the mask of.
Until cudatoolkit and CUDA_PATH issues get resolved, the program runs on cpu only mode, so specify it in the respective flag. If device = "cuda", follwing error happnes:
"NameError: name '_C' is not defined"

```bash
python groundedsam_func.py   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py   --grounded_checkpoint models/groundingdino_swint_ogc.pth   --sam_checkpoint models/sam_vit_h_4b8939.pth   --input_image inputs/dog.jpg   --output_dir "outputs/grounded_sam/"   --box_threshold 0.3   --text_threshold 0.25   --text_prompt "dog"   --device "cpu"
```
### 4. GroundedSAM-based inpainting
#### Checkpoints:
same as above
###### Input Command Exmple:
Specify via Text Prompt the object you want to detect and the object you want to replace it with.
Until cudatoolkit and CUDA_PATH issues get resolved, the program runs on cpu only mode, so specify it in the respective flag. If device = "cuda", follwing error happnes:
"NameError: name '_C' is not defined"
```bash
 python groundedsam_inpaint.py   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py   --grounded_checkpoint models/groundingdino_swint_ogc.pth   --sam_checkpoint models/sam_vit_h_4b8939.pth   --input_image inputs/dog.jpg   --output_dir "outputs/grounded_sam"   --box_threshold 0.3   --text_threshold 0.25   --det_prompt "dog"   --inpaint_prompt "bear cub, high quality, detailed"   --device "cpu"
 ```

### 5. Using the gradio app for groundedSAM

```bash
cd Grounded-Segment-Anything
python gradio_app.py
```

For it to run properly, the following modifications were performed:

Lines 196/197: change "image" to "composite" and "mask" to "layers"
Line 372: input_image = gr.ImageEditor(sources='upload', type="pil", value="assets/demo2.jpg")
Line 376: run_button = gr.Button()
Lines 391-394: with gr.Column():
                gallery = gr.Gallery(
                label="Generated images", 
                show_label=False, 
                elem_id="gallery", 
                preview=True, 
                object_fit="scale-down"
                )
Line 399: just comment out or remove
Line 400 (optional): change share=True if you need a public link