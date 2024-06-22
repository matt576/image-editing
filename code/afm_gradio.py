import gradio as gr
from PIL import Image

# imports from code scripts
from mask_func import sam_gradio, sam_gradio_text
from inpaint_func import controlnet_inpaint_gradio
from inpaint_ldm import ldm_removal_gradio
from superres_func import superres_gradio
from restyling_func import restyling_gradio
from groundedsam_func import groundedsam_mask_gradio
from groundedsam_inpaint import groundedsam_inpaint_gradio

selected_points = []

def run_afm_app(task_selector, input_image, mask_image, text_input, coord_input, ddim_steps, inpaint_input, coord_input_text):
    
    if task_selector == "SAM Mask Generation - Image Click Input": # mask_func.py
        return sam_gradio_text(input_image, coord_input)

    if task_selector == "SAM Mask Generation - Text Input": # e.g. "350,500; 450,500; 490,500"
        return sam_gradio_text(input_image, coord_input_text)

    if task_selector == "ControlNet Inpainting": # inpaint_func.py
        return controlnet_inpaint_gradio(input_image, mask_image, text_input)

    if task_selector == "Object Removal": # inpaint_ldm.py
        return ldm_removal_gradio(input_image, mask_image, ddim_steps)
    
    if task_selector == "Restyling":
        return restyling_gradio(input_image, text_input)

    if task_selector == "Hyperresolution":
        return superres_gradio(input_image)

    if task_selector == "GroundedSAM Mask Generation":
        return groundedsam_mask_gradio(input_image, text_input)

    if task_selector == "GroundedSAM Inpainting":
        return groundedsam_inpaint_gradio(input_image, text_input, inpaint_input)

def input_handler(evt: gr.SelectData):
    # # print(evt.__dict__)
    # coords = evt.index  # Get the coordinates from the event data
    # x, y = coords[0], coords[1]
    # coord_string = f"{x},{y}"
    # return coord_string
    global selected_points
    coords = evt.index  # Get the coordinates from the event data
    x, y = coords[0], coords[1]
    selected_points.append([x, y])
    print(f"Selected points: {selected_points}") 
    # Format points as a semicolon-separated string
    coord_string = '; '.join([f"{pt[0]},{pt[1]}" for pt in selected_points])
    print("coord_string: ", coord_string)
    return coord_string

def reset_selected_points(input_image):
    global selected_points
    selected_points = []  # Reset the list when a new image is loaded
    print("Selected points have been reset.")  # Debug statement
    return None

if __name__ == "__main__":
    
    block = gr.Blocks()

    with block:
        with gr.Row():
            with gr.Column():

                task_selector = gr.Dropdown(["SAM Mask Generation - Text Input",
                                            "SAM Mask Generation - Image Click Input", 
                                            "ControlNet Inpainting", 
                                            "Object Removal",
                                            "Restyling",
                                            "Hyperresolution",
                                            "GroundedSAM Mask Generation",
                                            "GroundedSAM Inpainting"], 
                                            value="SAM Mask Generation - Text Input")
                input_image = gr.Image(label="Raw Input Image", sources='upload', type="pil", value="inputs/car.png", interactive=True)
                # input_image.select(input_handler)
                coord_input_text = gr.Textbox(label="Pixel Coordinates (x,y) from text input, Example: 350,500; 450,500; 490,500 ", value="350,500; 450,500; 490,500") # for SAM
                coord_input = gr.Textbox(label="Pixel Coordinates (x,y) from clicking image", value="") # for SAM
                mask_image = gr.Image(label="Input Mask (Optional)", sources='upload', type="pil", value="inputs/batman_mask.jpg")
                text_input = gr.Textbox(label="Text Prompt: ControlNet inpaint or GroundedSAM object mask", value="") # for inpainting or restyling
                ddim_steps = gr.Textbox(label="Number of DDIM sampling steps for object removal", value="50") # for inpaint_ldm
                inpaint_input = gr.Textbox(label="GroundedSAM Inpainting Prompt", value="") # for groundedSAM inpainting
                
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
                reset_button = gr.Button("Reset Points")

        input_image.select(input_handler, inputs=[], outputs=coord_input) # clicking input for sam image coordinates
        input_image.change(reset_selected_points, inputs=[input_image], outputs=[])

        generate_button.click(
                fn = run_afm_app,
                inputs=[task_selector, input_image, mask_image, text_input, coord_input, ddim_steps, inpaint_input, coord_input_text],
                outputs = output_image
        )

        reset_button.click(
                fn = reset_selected_points,
                inputs = [input_image],
                outputs = None
        )  

    block.launch(share=True)