import gradio as gr
from PIL import Image, ImageDraw

# imports from code scripts
from mask_func import sam_gradio
from inpaint_func import controlnet_inpaint_gradio
from inpaint_ldm import ldm_removal_gradio
from superres_func import superres_gradio
from restyling_func import restyling_gradio
from groundedsam_func import groundedsam_mask_gradio
from groundedsam_inpaint import groundedsam_inpaint_gradio
from inpaint_sd import inpaint_sd_gradio
from inpaint_sdxl import inpaint_sdxl_gradio
from inpaint_kandinsky import inpaint_kandinsky_gradio

def run_afm_app(task_selector, input_image, mask_image, text_input, text_input_x, coord_input, ddim_steps, inpaint_input):
    print(f"Task selected: {task_selector}")
    if task_selector == "SAM": # mask_func.py
        return sam_gradio(input_image, coord_input)

    if task_selector == "GroundedSAM":
        return groundedsam_mask_gradio(input_image, text_input)

    if task_selector == "Stable Diffusion with ControlNet Inpainting": # inpaint_func.py
        return controlnet_inpaint_gradio(input_image, mask_image, text_input_x)

    if task_selector == "Stable Diffusion": # inpaint_sd.py
        print("here", text_input_x)
        return inpaint_sd_gradio(input_image, mask_image, text_input_x)
    
    if task_selector == "Stable Diffusion XL": # inpaint_sdxl.py
        print("heree",text_input_x)
        return inpaint_sdxl_gradio(input_image, mask_image, text_input_x)

    if task_selector == "Kandinsky v2.2": # inpaint_kandinsky.py
        print("hereee",text_input_x)
        return inpaint_kandinsky_gradio(input_image, mask_image, text_input_x)

    if task_selector == "GroundedSAM Inpainting":
        return groundedsam_inpaint_gradio(input_image, text_input, inpaint_input)

    if task_selector == "Object Removal LDM": # inpaint_ldm.py
        return ldm_removal_gradio(input_image, mask_image, ddim_steps)
    
    if task_selector == "Restyling - Stable Diffusion v1.5":
        return restyling_gradio(input_image, text_input)

    if task_selector == "Superresolution - Stable Diffusion v1.5":
        return superres_gradio(input_image)

selected_points = []
#testtets
def input_handler(evt: gr.SelectData, input_image):
    global selected_points
    coords = evt.index
    x, y = coords[0], coords[1]
    selected_points.append([x, y])
    coord_string = '; '.join([f"{pt[0]},{pt[1]}" for pt in selected_points])

    image_with_points = input_image.copy()
    draw = ImageDraw.Draw(image_with_points)
    for point in selected_points:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill="red", outline="red")

    return coord_string, image_with_points

def reset_selected_points(input_image):
    global selected_points
    selected_points = []
    print("Selected points have been reset.")
    return "", input_image

title = "# AFM Image-Editing App"
if __name__ == "__main__":
    
    block = gr.Blocks(theme='shivi/calm_seafoam')

    with block:
        gr.Markdown(title)
        gr.Markdown(
        """
        Welcome to the AFM Image-Editing App!
        First select the desired task on the Dropdown menu below.
        Then, input the necessary prompts.
        Finally, click on 'Generate' and enjoy the App!
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Raw Input Image", sources='upload', type="pil", value="test_dataset/jessi.png", interactive=True)

            with gr.Column():
                task_selector = gr.Dropdown(["SAM", "GroundedSAM",
                                            "Stable Diffusion with ControlNet Inpainting",
                                            "Stable Diffusion",
                                            "Stable Diffusion XL",
                                            "Kandinsky v2.2",
                                            "GroundedSAM Inpainting",
                                            "Object Removal LDM",
                                            "Restyling - Stable Diffusion v1.5",
                                            "Superresolution - Stable Diffusion v1.5"],
                                            label="Select Model")
                with gr.Tab("Mask Generation"):
                    # task_selector = gr.Dropdown(["SAM", 
                    #                             "GroundedSAM"], 
                    #                             label="Select Model")

                    gr.Markdown("SAM")
                    gr.Markdown("Required inouts: Pixel Coordinates")
                    gr.Markdown("Type image coordinates manually or click on the image directly.")
                    coord_input = gr.Textbox(label="Pixel Coordinates (x,y), Format x1,y1; x2,y2 ...", value="") # for SAM
                    reset_button = gr.Button("Reset Points")

                    gr.Markdown("GroundedSAM")
                    gr.Markdown("Required Inputs: Text Promt - Object to be masked")
                    gr.Markdown("Input in the text box below the object(s) in the input image for which the masks are to be generated")
                    text_input = gr.Textbox(label="Text Prompt: ")

                with gr.Tab("Inpainting - Object Replacement"):
                    gr.Markdown("Object Replacement with own uploaded input mask")
                    # task_selector = gr.Dropdown(["Stable Diffusion with ControlNet Inpainting",
                    #                             "Stable Diffusion",
                    #                             "Stable Diffusion XL",
                    #                             "Kandinsky v2.2"],
                    #                             label="Select Model")

                    gr.Markdown("Input in the text box below the desired object to be inpainted in place of the mask input below.")
                    gr.Markdown("Example prompt: white tiger, photorealistic, detailed, high quality")
                    text_input_x = gr.Textbox(label="Text Prompt: ")
                
                with gr.Tab("Inpainting Pipelines"):
                    gr.Markdown("Object Replacement with input photo only")
                    # task_selector = gr.Dropdown(["GroundedSAM Inpainting"],
                    #                             label="Select Model")

                    gr.Markdown("GroundedSAM Inpainting")
                    gr.Markdown("First, type which object to be detected, then what is going to replace it.")
                    text_input = gr.Textbox(label="Text Prompt: ")
                    inpaint_input = gr.Textbox(label="Inpainting Prompt: ")

                with gr.Tab("Object Removal"):
                    gr.Markdown("Object Removal with own uploaded input mask")
                    # task_selector = gr.Dropdown(["Object Removal LDM"],
                    #                             label="Select Model")

                    gr.Markdown("Object Removal Latent Diffusion Model")
                    gr.Markdown("Given the uploaded mask below, the correspondent object will be removed and the empty space completed.")
                    gr.Markdown("Choose the desired number of DDDIM steps below:")
                    ddim_steps = gr.Slider(minimum=5, maximum=200, label="Number of DDIM sampling steps for object removal", value=150)

                with gr.Tab("Object Removal Pipelines"):
                    gr.Markdown("Object Replacement with input photo only")
                    # task_selector = gr.Dropdown(["Object Removal LDM"],
                    #                             label="Select Model")

                    gr.Markdown("GroundedSAM Inpainting")
                    gr.Markdown("First, type which object to be detected, then what is going to replace it.")
                    text_input = gr.Textbox(label="Text Prompt: ")
                    inpaint_input = gr.Textbox(label="Inpainting Prompt: ")
                
                with gr.Tab("Restyling"):
                    # task_selector = gr.Dropdown(["Restyling - Stable Diffusion v1.5"],
                    #                             label="Select Model")
                    text_input = gr.Textbox(label="Text Prompt: ")

                with gr.Tab("Superresolution"):
                    # task_selector = gr.Dropdown(["Superresolution - Stable Diffusion v1.5"],
                    #                             label="Select Model")
                    gr.Markdown("Select model on the Dropdown menu and simply click the 'Generate' button to get your new image.")
        
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Mask Input", open=False):
                    mask_image = gr.Image(label="Input Mask", sources='upload', type="pil", value="inputs/ldm_inputs/jessi_mask.png")

            with gr.Column():
                generate_button = gr.Button("Generate")
                output_image = gr.Image(label="Generated Image", type="pil")
                                  
        
        input_image.select(input_handler, inputs=[input_image], outputs=[coord_input, input_image]) # clicking input for sam image coordinates
        # input_image.change(reset_selected_points, inputs=[input_image], outputs=[coord_input])

        generate_button.click(
                fn = run_afm_app,
                inputs=[task_selector, input_image, mask_image, text_input, text_input_x, coord_input, ddim_steps, inpaint_input],
                outputs = output_image
        )

        reset_button.click(
                fn = reset_selected_points,
                inputs = [input_image],
                outputs = [coord_input, input_image]
        )  

    block.launch(share=True)