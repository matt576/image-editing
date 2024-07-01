import gradio as gr
from PIL import Image, ImageDraw

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
from ldm_removal_pipe import ldm_removal_pipe_gradio
from inpaint_pipe import inpaint_pipe_gradio
from inpaint_func_pipe import inpaint_func_pipe_gradio

def run_afm_app(task_selector, input_image, mask_image, text_input, text_input_x, text_input_gsam, coord_input, 
                ddim_steps, ddim_steps_pipe, inpaint_input_gsam, text_input_inpaint_pipe):
    print(f"Task selected: {task_selector}")

    if task_selector == "SAM":
        return sam_gradio(input_image, coord_input)

    if task_selector == "GroundedSAM":
        return groundedsam_mask_gradio(input_image, text_input)

    if task_selector == "Stable Diffusion with ControlNet Inpainting":
        return controlnet_inpaint_gradio(input_image, mask_image, text_input_x)

    if task_selector == "Stable Diffusion v1.5 Inpainting":
        return inpaint_sd_gradio(input_image, mask_image, text_input_x)

    if task_selector == "Stable Diffusion XL Inpainting":
        return inpaint_sdxl_gradio(input_image, mask_image, text_input_x)

    if task_selector == "Kandinsky v2.2 Inpainting":
        return inpaint_kandinsky_gradio(input_image, mask_image, text_input_x)

    if task_selector == "GroundedSAM Inpainting":
        return groundedsam_inpaint_gradio(input_image, text_input_gsam, inpaint_input_gsam)

    if task_selector == "Object Removal LDM":
        return ldm_removal_gradio(input_image, mask_image, ddim_steps)

    if task_selector == "Restyling - Stable Diffusion v1.5":
        return restyling_gradio(input_image, text_input)

    if task_selector == "Superresolution - Stable Diffusion v1.5":
        return superres_gradio(input_image)

    if task_selector == "LDM Removal Pipeline":
        return ldm_removal_pipe_gradio(input_image, coord_input, ddim_steps_pipe)

    if task_selector in ["Stable Diffusion v1.5 Inpainting Pipeline", "Stable Diffusion XL Inpainting Pipeline", "Kandinsky v2.2 Inpainting Pipeline"]:
        return inpaint_pipe_gradio(task_selector, input_image, coord_input, text_input_inpaint_pipe)

    if task_selector == "Stable Diffusion with ControlNet Inpainting Pipeline":
        return inpaint_func_pipe_gradio(input_image, coord_input, text_input_inpaint_pipe)

selected_points = []

def input_handler(evt: gr.SelectData, input_image):
    global selected_points
    coords = evt.index
    x, y = coords[0], coords[1]
    selected_points.append([x, y])
    coord_string = '; '.join([f"{pt[0]},{pt[1]}" for pt in selected_points])

    image_with_points = input_image.copy()
    draw = ImageDraw.Draw(image_with_points)
    for point in selected_points:
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="red", outline="red")

    return coord_string, image_with_points

def reset_selected_points(input_image):
    global selected_points
    selected_points = []
    print("Selected points have been reset.")
    return "", input_image

def reload_image(original_image_path):
    original_image = original_image_path
    return original_image

def update_task_selector(task_selector, task):
    return task

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

        original_image_path = "test_dataset/jessi.png"
        original_image = Image.open(original_image_path)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", sources='upload', type="pil", value=original_image_path, interactive=True)
                # gr.Markdown("Type image coordinates manually or click on the image directly:")
                # coord_input = gr.Textbox(label="Pixel Coordinates (x,y), Format x1,y1; x2,y2 ...", value="")
                # reset_button = gr.Button("Reset Points")
                # reload_image_button = gr.Button("Reload Original Image without points")

            with gr.Column():
                gr.Markdown("Type image coordinates manually or click on the image directly:")
                coord_input = gr.Textbox(label="Pixel Coordinates (x,y), Format x1,y1; x2,y2 ...", value="")
                reset_button = gr.Button("Reset coordinates")
                reload_image_button = gr.Button("Reload Original Image")
                task_selector = gr.State(value="")

                with gr.Tab("Mask Generation Preview"):
                    tab_task_selector_1 = gr.Dropdown(["SAM", "GroundedSAM"], label="Select Model")
                    gr.Markdown("""
                                ### Instructions
                                - **SAM**:  
                                Required inputs: Pixel Coordinates  
                                Type image coordinates manually or click on the image directly. Finally, simply click on the 'Generate' button.
                                - **GroundedSAM (GroundingDINO + SAM)**:  
                                Required Inputs: Text Prompt - Object to be masked  
                                Input in the text box below the object(s) in the input image for which the masks are to be generated.
                                """)
                    text_input = gr.Textbox(label="Text Prompt: ")

                with gr.Tab("Inpainting - Object Replacement"):
                    tab_task_selector_2 = gr.Dropdown(["Stable Diffusion with ControlNet Inpainting", 
                                                    "Stable Diffusion v1.5 Inpainting", 
                                                    "Stable Diffusion XL Inpainting", 
                                                    "Kandinsky v2.2 Inpainting"], 
                                                    label="Select Model")
                    gr.Markdown("""
                                ### Instructions
                                All models in this section work with the given uploaded input mask.  
                                Required Inputs: Input Mask (Upload) , Text Prompt - Object to replace masked area on given input mask below.  
                                Input in the text box below the desired object to be inpainted in place of the mask input below.  
                                Example prompt: 'white tiger, photorealistic, detailed, high quality'.
                                """)
                    text_input_x = gr.Textbox(label="Text Prompt: ")

                with gr.Tab("Object Removal"):
                    tab_task_selector_3 = gr.Dropdown(["Object Removal LDM"], label="Select Model")
                    gr.Markdown("""
                                ### Instructions
                                - **Object Removal LDM**:  
                                Required inputs: Input Mask (Upload) , DDIM Steps  
                                Given the uploaded mask below, simply adjust the slider below according to the desired number of iterations:
                                """)
                    ddim_steps = gr.Slider(minimum=5, maximum=200, label="Number of DDIM sampling steps for object removal", value=150)

                with gr.Tab("Restyling"):
                    tab_task_selector_4 = gr.Dropdown(["Restyling - Stable Diffusion v1.5"], label="Select Model")
                    text_input = gr.Textbox(label="Text Prompt: ")

                with gr.Tab("Superresolution"):
                    tab_task_selector_5 = gr.Dropdown(["Superresolution - Stable Diffusion v1.5"], label="Select Model")
                    gr.Markdown("Select model on the Dropdown menu and simply click the 'Generate' button to get your new image.")

                with gr.Tab("Pipeline: Inpainting - Object Replacement"):
                    tab_task_selector_6 = gr.Dropdown(["GroundedSAM Inpainting",
                                                    "Stable Diffusion with ControlNet Inpainting Pipeline", 
                                                    "Stable Diffusion v1.5 Inpainting Pipeline", 
                                                    "Stable Diffusion XL Inpainting Pipeline", 
                                                    "Kandinsky v2.2 Inpainting Pipeline"], label="Select Model")
                    gr.Markdown("""
                                - **GroundedSAM Inpainting (GroundingDINO + SAM + Stable Diffusion)**:  
                                Required Inputs: Detection Prompt , Inpainting Prompt  
                                Input in the text box below the object(s) in the input image for which the masks are to be generated.  
                                Example detection prompt: 'dog'.  
                                Example inpaint prompt: 'white tiger, photorealistic, detailed, high quality'.
                                """)
                    text_input_gsam = gr.Textbox(label="Detection Prompt: ")
                    inpaint_input_gsam = gr.Textbox(label="Inpainting Prompt: ")
                    gr.Markdown("""
                                - **Kandinsky v2.2 / Stable Diffusion v1.5 / SDXL / SD + ControlNet**:  
                                Required Inputs: Pixel Coodinates , Inpainting Prompt  
                                Input in the text box below the object(s) in the input image for which the masks are to be generated.  
                                Example prompt: 'white tiger, photorealistic, detailed, high quality'.
                                """)
                    text_input_inpaint_pipe = gr.Textbox(label="Text Prompt: ")

                with gr.Tab("Pipeline - Object Removal"):
                    tab_task_selector_7 = gr.Dropdown(["LDM Removal Pipeline"], label="Select Model")
                    gr.Markdown("""
                                ### Instructions
                                - **LDM Removal Pipeline**:  
                                Required inputs: Pixel Coodinates, DDIM Steps  
                                If you wish to view the mask before the fnal output, go to the 'Mask Generation Preview' Tab.  
                                Type the image coordinates manually in the box under the image or click on the image directly.  
                                For a more detailed mask of a specific object or part of it, select multiple points.  
                                Finally, choose number of DDIM steps simply click on the 'Generate' button:
                                """)
                    ddim_steps_pipe = gr.Slider(minimum=5, maximum=200, label="Number of DDIM sampling steps for object removal", value=150)
                
                
                
                generate_button = gr.Button("Generate")

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Mask Input (Optional)", open=False):
                    mask_image = gr.Image(label="Input Mask (Optional)", sources='upload', type="pil", value="inputs/ldm_inputs/jessi_mask.png")

            with gr.Column():
                # generate_button = gr.Button("Generate")
                output_image = gr.Image(label="Generated Image", type="pil")

        input_image.select(input_handler, inputs=[input_image], outputs=[coord_input, input_image])

        generate_button.click(
            fn=run_afm_app,
            inputs=[task_selector, input_image, mask_image, text_input, text_input_x, text_input_gsam, coord_input, ddim_steps, ddim_steps_pipe, inpaint_input_gsam, text_input_inpaint_pipe],
            outputs=output_image
        )

        reset_button.click(
            fn=reset_selected_points,
            inputs=[input_image],
            outputs=[coord_input, input_image]
        )

        reload_image_button.click(
            fn=reload_image,
            inputs=[gr.State(original_image_path)],
            outputs=[input_image]
        )

        tab_task_selector_1.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_1], outputs=[task_selector])
        tab_task_selector_2.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_2], outputs=[task_selector])
        tab_task_selector_3.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_3], outputs=[task_selector])
        tab_task_selector_4.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_4], outputs=[task_selector])
        tab_task_selector_5.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_5], outputs=[task_selector])
        tab_task_selector_6.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_6], outputs=[task_selector])
        tab_task_selector_7.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_7], outputs=[task_selector])

    block.launch(share=True)