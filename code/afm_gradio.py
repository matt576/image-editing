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

def run_afm_app(task_selector, input_image, mask_image, text_input, coord_input, ddim_steps, inpaint_input):
    
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

    if task_selector == "GroundedSAM Mask Generation":
        return groundedsam_mask_gradio(input_image, text_input)

    if task_selector == "GroundedSAM Inpainting":
        return groundedsam_inpaint_gradio(input_image, text_input, inpaint_input)

selected_points = []
#testtets
def input_handler(evt: gr.SelectData, input_image):
    # # print(evt.__dict__)
    # coords = evt.index
    # x, y = coords[0], coords[1]
    # coord_string = f"{x},{y}"
    # return coord_string

    global selected_points
    coords = evt.index
    x, y = coords[0], coords[1]
    selected_points.append([x, y])
    coord_string = '; '.join([f"{pt[0]},{pt[1]}" for pt in selected_points]) # Format points as a semicolon-separated string
    # print(f"Selected points: {selected_points}")
    # print("coord_string: ", coord_string)

    # Draw points on the image
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

                task_selector = gr.Dropdown(["SAM Mask Generation", 
                                            "ControlNet Inpainting", 
                                            "Object Removal",
                                            "Restyling",
                                            "Hyperresolution",
                                            "GroundedSAM Mask Generation",
                                            "GroundedSAM Inpainting"], 
                                            value="SAM Mask Generation")

                with gr.Tab("SAM Mask Generation"):
                        gr.Markdown("Type image coordinates manually or click on the image directly.")
                        coord_input = gr.Textbox(label="Pixel Coordinates (x,y), Format x1,y1; x2,y2 ...", value="") # for SAM
                        reset_button = gr.Button("Reset Points")

                with gr.Tab("ControlNet Inpainting"):
                    gr.Markdown("Input in the text box below the desired object to be inpainted in place of the mask input above.")
                    text_input = gr.Textbox(label="Text Prompt: ", value="")

                with gr.Tab("Object Removal"):
                    ddim_steps = gr.Textbox(label="Number of DDIM sampling steps for object removal", value="50") # for inpaint_ldm

                with gr.Tab("GroundedSAM Mask Generation"):
                    gr.Markdown("Input in the text box below the object(s) in the input image for which the masks are to be generated")
                    text_input = gr.Textbox(label="Text Prompt: ", value="")

                with gr.Tab("GroundedSAM Inpainting"):
                    gr.Markdown("First, type which object to be detected, then what is going to replace it.")
                    text_input = gr.Textbox(label="Text Prompt: ", value="")
                    inpaint_input = gr.Textbox(label="Inpainting Prompt: ", value="")
                
                with gr.Tab("Restyling"):
                    text_input = gr.Textbox(label="Text Prompt: ", value="")

                with gr.Tab("Hyperresolution"):
                    gr.Markdown("Simply click the Generate button above to get your new image.")


                generate_button = gr.Button("Generate")

        
        with gr.Row():
                with gr.Column():
                    with gr.Accordion("Mask Input", open=False):
                        mask_image = gr.Image(label="Input Mask", sources='upload', type="pil", value="inputs/ldm_inputs/jessi_mask.png")

                with gr.Column():
                    output_image = gr.Image(label="Generated Image", type="pil")
                                  
                
        # output_image = gr.Image(label="Generated Image", type="pil")

        
        input_image.select(input_handler, inputs=[input_image], outputs=[coord_input, input_image]) # clicking input for sam image coordinates
        # input_image.change(reset_selected_points, inputs=[input_image], outputs=[coord_input])

        generate_button.click(
                fn = run_afm_app,
                inputs=[task_selector, input_image, mask_image, text_input, coord_input, ddim_steps, inpaint_input],
                outputs = output_image
        )

        reset_button.click(
                fn = reset_selected_points,
                inputs = [input_image],
                outputs = [coord_input, input_image]
        )  

    block.launch(share=True)