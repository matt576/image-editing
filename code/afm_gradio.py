import gradio as gr
from PIL import Image

# imports from code scripts
from mask_func import sam_gradio
from inpaint_func import controlnet_inpaint_gradio
from inpaint_ldm import ldm_removal_gradio
from superres_func import superres_gradio
from restyling_func import restyling_gradio

def run_afm_app(task_selector, input_image, mask_image, text_input, coord_input, ddim_steps):
    
    if task_selector == "SAM Mask Generation": # mask_func.py
        return sam_gradio(input_image, coord_input)

    if task_selector == "ControlNet Inpainting": # inpaint_func.py
        return controlnet_inpaint_gradio(input_image, mask_image, text_input)

    if task_selector == "Object Removal": # inpaint_ldm.py
        return ldm_removal_gradio(input_image, mask_image, ddim_steps)
    
    if task_selector == "Restyling":
        return restyling_gradio(input_image, text_input)

    if task_selector == "Hyperresolution":
        return superres_gradio(input_image)

def input_handler(evt: gr.SelectData):
    # print(evt.__dict__)
    coords = evt.index  # Get the coordinates from the event data
    x, y = coords[0], coords[1]
    coord_string = f"{x},{y}"
    return coord_string

if __name__ == "__main__":
    
    block = gr.Blocks()

    with block:
        with gr.Row():
            with gr.Column():

                task_selector = gr.Dropdown(["SAM Mask Generation", 
                                            "ControlNet Inpainting", 
                                            "Object Removal",
                                            "Restyling",
                                            "Hyperresolution"], 
                                            value="SAM Mask Generation")
                input_image = gr.Image(label="Raw Input Image", sources='upload', type="pil", value="inputs/batman.jpg", interactive=True)
                input_image.select(input_handler)
                coord_input = gr.Textbox(label="Pixel Coordinates (x,y)", value="350,500") # for SAM
                mask_image = gr.Image(label="Input Mask (Optional)", sources='upload', type="pil", value="inputs/batman_mask.jpg")
                text_input = gr.Textbox(label="Text Prompt", value="") # for inpainting or restyling
                ddim_steps = gr.Textbox(label="Number of DDIM sampling steps for object removal", value="50") # for inpaint_ldm
                
            with gr.Column():
                # gallery = gr.Gallery(
                # label="Generated images", 
                # show_label=False, 
                # elem_id="gallery", 
                # preview=True, 
                # object_fit="scale-down"
                # )
                output_image = gr.Image(label="Generated Image", type="pil")
                generate_button = gr.Button("Generate")

        input_image.select(input_handler, inputs=[], outputs=coord_input) # clicking input for sam image coordinates
        generate_button.click(
                fn = run_afm_app,
                inputs=[task_selector, input_image, mask_image, text_input, coord_input, ddim_steps],
                outputs = output_image
        )  

    block.launch(share=True)