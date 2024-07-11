import gradio as gr
from PIL import Image, ImageDraw


def run_afm_app(task_selector, input_image, mask_image, text_input, text_input_x, text_input_gsam, coord_input, 
                ddim_steps, ddim_steps_pipe, inpaint_input_gsam, text_input_inpaint_pipe, text_input_restyling,
                blur, sharpen, prompt_outpaint, e_l, e_r, e_u, e_d, steps_outpaint, prompt_background , steps_br,
                str_res, gs_res, np_res, steps_res, np_inpaint, steps_inpaint, prompt_txt2img, np_txt2img, gs_txt2img, 
                steps_txt2img, steps_super, dilation_bool, dilation_value, steps_inp):

    print(f"Task selected: {task_selector}")

    if task_selector == "SAM":
        from mask_sam import sam_gradio
        return sam_gradio(input_image, coord_input, dilation_bool, dilation_value)

    if task_selector == "GroundedSAM":
        from mask_groundedsam import groundedsam_mask_gradio
        return groundedsam_mask_gradio(input_image, text_input, dilation_bool, dilation_value)

    if task_selector == "Stable Diffusion with ControlNet Inpainting":
        from inpaint_sd_controlnet import controlnet_inpaint_gradio
        return controlnet_inpaint_gradio(input_image, mask_image, text_input_x)

    if task_selector == "Stable Diffusion v1.5 Inpainting":
        from inpaint_sd import inpaint_sd_gradio
        return inpaint_sd_gradio(input_image, mask_image, text_input_x, steps_inp)

    if task_selector == "Stable Diffusion XL Inpainting":
        from inpaint_sdxl import inpaint_sdxl_gradio
        return inpaint_sdxl_gradio(input_image, mask_image, text_input_x, steps_inp)

    if task_selector == "Kandinsky v2.2 Inpainting":
        from inpaint_kandinsky import inpaint_kandinsky_gradio
        return inpaint_kandinsky_gradio(input_image, mask_image, text_input_x, steps_inp)

    if task_selector == "GroundedSAM Inpainting":
        from inpaint_groundedsam import groundedsam_inpaint_gradio
        return groundedsam_inpaint_gradio(input_image, text_input_gsam, inpaint_input_gsam)

    if task_selector == "Object Removal LDM":
        from eraser_ldm import ldm_removal_gradio
        return ldm_removal_gradio(input_image, mask_image, ddim_steps)

    if task_selector == "Restyling - Stable Diffusion v1.5":
        from restyling_sd import restyling_gradio
        return restyling_gradio(input_image, text_input_restyling, str_res, gs_res, np_res, steps_res)

    if task_selector == "Restyling - Stable Diffusion XL":
        from restyling_sdxl import restyling_sdxl_gradio
        return restyling_sdxl_gradio(input_image, text_input_restyling, str_res, gs_res, np_res, steps_res)

    if task_selector == "Restyling - Kandinsky v2.2":
        from restyling_kandinsky import restyling_kandinsky_gradio
        return restyling_kandinsky_gradio(input_image, text_input_restyling, str_res, gs_res, np_res, steps_res)

    if task_selector == "Superresolution - LDM x4 OpenImages":
        from superres_ldm import superres_gradio
        return superres_gradio(input_image, steps_super)

    if task_selector == "Superresolution - Stability AI x4 Upscaler":
        from superres_upscaler import superres_upscaler_gradio
        return superres_upscaler_gradio(input_image, steps_super)

    if task_selector == "LDM Removal Pipeline":
        from eraser_ldm_pipe import ldm_removal_pipe_gradio
        return ldm_removal_pipe_gradio(input_image, coord_input, ddim_steps_pipe)

    if task_selector in ["Stable Diffusion v1.5 Inpainting Pipeline", "Stable Diffusion XL Inpainting Pipeline", "Kandinsky v2.2 Inpainting Pipeline"]:
        from inpaint_pipe import inpaint_pipe_gradio
        return inpaint_pipe_gradio(task_selector, input_image, coord_input, text_input_inpaint_pipe, np_inpaint, steps_inpaint)

    if task_selector == "Stable Diffusion with ControlNet Inpainting Pipeline":
        from inpaint_sd_controlnet_pipe import inpaint_func_pipe_gradio
        return inpaint_func_pipe_gradio(input_image, coord_input, text_input_inpaint_pipe, np_inpaint, steps_inpaint)

    if task_selector == "Portrait Mode - Depth Anything":
        from blur_image import portrait_gradio
        return portrait_gradio(input_image, blur, sharpen)

    if task_selector == "Outpainting - Stable Diffusion":
        from outpaint_sd import outpaint_sd_gradio
        return outpaint_sd_gradio(input_image, prompt_outpaint, e_l, e_r, e_u, e_d, steps_outpaint)

    if task_selector == "Outpainting - Stable Diffusion XL":
        from outpaint_sdxl import outpaint_sdxl_gradio
        return outpaint_sdxl_gradio(input_image, prompt_outpaint, e_l, e_r, e_u, e_d, steps_outpaint)

    if task_selector == "Background Replacement - Stable Diffusion":
        from background_replace_sd import background_replace_sd_gradio
        return background_replace_sd_gradio(input_image, prompt_background , steps_br)

    if task_selector == "Background Replacement - Stable Diffusion XL":
        from background_replace_sdxl import background_replace_sdxl_gradio
        return background_replace_sdxl_gradio(input_image, prompt_background , steps_br)

    if task_selector in ["Stable Diffusion v1.5 Txt2Img", "Stable Diffusion XL Txt2Img", "Kandinsky v2.2 Txt2Img"]:
        from txt2img_generation import txt2img_gradio
        return txt2img_gradio(input_image, task_selector, prompt_txt2img, np_txt2img, gs_txt2img, steps_txt2img)

    if task_selector == "Eraser - LaMa":
        from eraser_lama import eraser_lama_gradio
        return eraser_lama_gradio(input_image, mask_image)

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

def reload_image_with_output(output_image):
    return output_image

def reload_mask(output_image):
    return output_image

title = "# AFM Image-Editing App"

if __name__ == "__main__":
    block = gr.Blocks(theme='shivi/calm_seafoam')

    with block:
        gr.Markdown(title)
        gr.Markdown(
        """
        Welcome to the AFM Image-Editing App!
        First, upload an input image or generate it via Txt2Img below.
        Then, choose the desired task by navigating the tabs.
        Finally, choose the model on the Dropdown within each tab and click on 'Generate'! Enjoy the App!
        """)

        original_image_path = "inputs/outpainting/scott.png" # Select input image path here
        # original_image_path = "outputs/txt2img/generated_input.png" # for txt2img generated input image
        input_mask_path = "outputs/sam/mask_gradio.png" # Optional, make sure it matches the input image
        original_image = Image.open(original_image_path)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", sources='upload', type="pil", value=original_image_path, interactive=True)
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
        
        with gr.Row():
            generate_button = gr.Button("Generate!")

        with gr.Row():
            with gr.Column():

                gr.Markdown("Type image coordinates manually or click on the image directly:")
                coord_input = gr.Textbox(label="Pixel Coordinates (x,y), Format x1,y1; x2,y2 ...", value="")
                reset_button = gr.Button("Reset coordinates")
                reload_image_button = gr.Button("Clear Image")
                reload_output_button = gr.Button("Load Output")
                task_selector = gr.State(value="")

                with gr.Accordion("Txt2Img Generation (Optional)", open=False):
                    tab_task_selector_11 = gr.Dropdown(["Stable Diffusion v1.5 Txt2Img",
                                                        "Stable Diffusion XL Txt2Img",
                                                        "Kandinsky v2.2 Txt2Img"], label="Select Model")
                    gr.Markdown("""
                                ### Instructions  
                                Use this feature if you wish to generate your own input image.  
                                After generation, simply uncomment the original_image_path line on the gradio script and relaunch the app!  
                                Required Inputs: Text Prompt, str_res, gs_res, np_res, steps_res  
                                Example prompt: "Photorealistic Gotham City night skyline, rain pouring down, dark clouds with streaks of lightning."
                                """)     
                    prompt_txt2img = gr.Textbox(label="Text Prompt: ")
                    np_txt2img = gr.Textbox(label="Negative Prompt", value="poor details, ugly, blurry")
                    gs_txt2img = gr.Slider(minimum=0.0, maximum=50.0, label="Guidance Scale", value=3.5)
                    steps_txt2img = gr.Slider(minimum=5, maximum=200, label="Number of inference steps", value=50, step=1)

                with gr.Accordion("Mask Input Tasks (Optional)", open=False):
                    gr.Markdown("""
                                Here is the mask uploaded directly from the gradio script, if you wish to change it,
                                use the Mask Generation Preview Tab and click the 'Load Preview Mask' button.
                                """)
                    mask_image = gr.Image(label="Input Mask (Optional)", sources='upload', type="pil", value=input_mask_path)

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
                        steps_inp = gr.Slider(minimum=5, maximum=200, label="Number of inference steps: ", value=50, step=1) 

                    with gr.Tab("Object Removal"):
                        tab_task_selector_3 = gr.Dropdown(["Object Removal LDM", "Eraser - LaMa"], label="Select Model")
                        gr.Markdown("""
                                    ### Instructions
                                    - **Object Removal LDM**:  
                                    Required inputs: Input image, Input Mask (Upload or from Preview), DDIM Steps  
                                    Given the uploaded mask below, simply adjust the slider below according to the desired number of iterations.
                                    - **Eraser - LaMa**:  
                                    Required inputs: Input image, Input Mask (Upload or from Preview)  
                                    Please note, due to compability issues with the LaMa model and our gradio app, the output visualiztion will not
                                    work in the app, but your output will be saved to: code/outputs/untracked/eraser-lama.
                                    """)
                        ddim_steps = gr.Slider(minimum=5, maximum=250, label="Number of DDIM sampling steps for object removal LDM", value=150, step=1)

            with gr.Column():

                with gr.Tab("Mask Generation Preview"):
                    tab_task_selector_1 = gr.Dropdown(["SAM", "GroundedSAM"], label="Select Model")
                    reload_mask_button = gr.Button("Load Preview Mask")
                    gr.Markdown("""
                                ### Instructions
                                - **SAM**:  
                                Required inputs: Input Image, Pixel Coordinates, (Optional) Dilation  
                                Type image coordinates manually or click on the image directly. Finally, simply click on the 'Generate' button.
                                """)
                    dilation_bool = gr.Dropdown(["Yes", "No"], label="Use dilation (recommended for inapinting)")
                    dilation_value = gr.Slider(minimum=0, maximum=50, label="Dilation value (recommended: 10) ", value=10, step = 1)
                    gr.Markdown("""
                                - **GroundedSAM (GroundingDINO + SAM)**:  
                                Required Inputs: Text Prompt [object(s) to be detected], (Optional) Dilation  
                                Input in the text box below the object(s) in the input image for which the masks are to be generated.
                                """)
                    text_input = gr.Textbox(label="Text Prompt: ")

                with gr.Tab("Restyling"):
                    tab_task_selector_4 = gr.Dropdown(["Restyling - Stable Diffusion v1.5",
                                                        "Restyling - Stable Diffusion XL",
                                                        "Restyling - Kandinsky v2.2"], label="Select Model")
                    gr.Markdown("""
                                ### Instructions
                                Required Inputs: Input Image, Text Prompt, str_res, gs_res, np_res, steps_res   
                                Example Text Prompt: "Photorealistic Gotham City night skyline, rain pouring down, dark clouds with streaks of lightning."  
                                Example Negative Prompt: "poor details, ugly, blurry"
                                """)     
                    text_input_restyling = gr.Textbox(label="Text Prompt: ")
                    str_res = gr.Slider(minimum=0.1, maximum=1.0, label="Strength: ", value=0.75)
                    gs_res = gr.Slider(minimum=0.0, maximum=50.0, label="Guidance Scale: ", value=7.5)
                    np_res = gr.Textbox(label="Negative Prompt: ")
                    steps_res = gr.Slider(minimum=5, maximum=150, label="Number of inference steps: ", value=30, step=1)

                with gr.Tab("Superresolution"):
                    tab_task_selector_5 = gr.Dropdown(["Superresolution - LDM x4 OpenImages",
                                                    "Superresolution - Stability AI x4 Upscaler"], label="Select Model")
                    gr.Markdown("""
                                ### Instructions  
                                Required Inputs: Input Image, Number of Inference Steps  
                                Select model on the Dropdown menu, number of inference steps, and click the 'Generate' button to get your new image.
                                """)
                    steps_super = gr.Slider(minimum=5, maximum=150, label="Number of inference steps: ", value=30, step=1)

                with gr.Tab("Pipeline: Inpainting - Object Replacement"):
                    tab_task_selector_6 = gr.Dropdown(["GroundedSAM Inpainting",
                                                    "Stable Diffusion with ControlNet Inpainting Pipeline", 
                                                    "Stable Diffusion v1.5 Inpainting Pipeline", 
                                                    "Stable Diffusion XL Inpainting Pipeline", 
                                                    "Kandinsky v2.2 Inpainting Pipeline"], label="Select Model")
                    gr.Markdown("""
                                - **GroundedSAM Inpainting (GroundingDINO + SAM + Stable Diffusion)**:  
                                Required Inputs: Input Image, Detection Prompt , Inpainting Prompt  
                                Input in the text box below the object(s) in the input image for which the masks are to be generated.  
                                Example detection prompt: 'dog'.  
                                Example inpaint prompt: 'white tiger, photorealistic, detailed, high quality'.
                                """)
                    text_input_gsam = gr.Textbox(label="Detection Prompt: ")
                    inpaint_input_gsam = gr.Textbox(label="Inpainting Prompt: ")
                    gr.Markdown("""
                                - **Kandinsky v2.2 / Stable Diffusion v1.5 / SDXL / SD + ControlNet**:  
                                Required Inputs: Input Image, Pixel Coodinates , Inpainting Prompt  
                                Input in the text box below the object(s) in the input image for which the masks are to be generated.  
                                Example Text Prompt: 'white tiger, photorealistic, detailed, high quality'.  
                                Example Negative Prompt: 'poor details, ugly, blurry'.
                                """)
                    text_input_inpaint_pipe = gr.Textbox(label="Text Prompt: ")
                    np_inpaint = gr.Textbox(label="Negative Prompt: ")
                    steps_inpaint = gr.Slider(minimum=5, maximum=200, label="Number of inference steps: ", value=150, step=1)                   

                with gr.Tab("Pipeline - Object Removal"):
                    tab_task_selector_7 = gr.Dropdown(["LDM Removal Pipeline", " "], label="Select Model")
                    gr.Markdown("""
                                ### Instructions
                                - **LDM Removal Pipeline**:  
                                Required inputs: Input Image, Pixel Coodinates, DDIM Steps  
                                If you wish to view the mask before the fnal output, go to the 'Mask Generation Preview' Tab.  
                                Type the image coordinates manually in the box under the image or click on the image directly.  
                                For a more detailed mask of a specific object or part of it, select multiple points.  
                                Finally, choose number of DDIM steps simply click on the 'Generate' button:
                                """)
                    ddim_steps_pipe = gr.Slider(minimum=5, maximum=250, label="Number of DDIM sampling steps for object removal", value=150, step=1)

                with gr.Tab("Portrait Mode"):
                    tab_task_selector_8 = gr.Dropdown(["Portrait Mode - Depth Anything"], label='Select Model')
                    gr.Markdown("""
                                ### Instructions
                                - **Portrait Mode - Depth Anything**:  
                                Required inputs: Input Image, box blur, sharpen  
                                Recommended blur values range: 15-25  
                                Recommended sharpen values range: 0-5   
                                Adjust the required inputs with the siders below:
                                """)
                    blur = gr.Slider(minimum=0, maximum=50, label="Box Blur value", value=15, step=1)
                    sharpen = gr.Slider(minimum=0, maximum=20, label="Sharpen Parameter", value=0, step=1)

                with gr.Tab("Outpainting"):
                    tab_task_selector_9 = gr.Dropdown(["Outpainting - Stable Diffusion", "Outpainting - Stable Diffusion XL"], label='Select Model')
                    gr.Markdown("""
                                ### Instructions
                                - **Outpainting - Stable Diffusion**:  
                                Required inputs: Input Image, Text Prompt, extend left/right/up/down, steps  
                                Choose how much and which direction you want to extend /outpaint your image and specify a text prompt.  
                                Example prompt: "dog in a park with other dogs and cats and deers, colorful flowers, blue sky, sunny weather, no clouds"
                                """)
                    prompt_outpaint = gr.Textbox(label="Text Prompt: ")
                    e_l = gr.Slider(minimum=0, maximum=1000, label="Extend Left", value=0, step=1)
                    e_r = gr.Slider(minimum=0, maximum=1000, label="Extend Right", value=200, step=1)
                    e_u = gr.Slider(minimum=0, maximum=1000, label="Extend Up", value=0, step=1)
                    e_d = gr.Slider(minimum=0, maximum=1000, label="Extend Down", value=0, step=1)
                    steps_outpaint = gr.Slider(minimum=0, maximum=200, label="Number of Steps", value=50, step=1)

                with gr.Tab("Background Replacement"):
                    tab_task_selector_10 = gr.Dropdown(["Background Replacement - Stable Diffusion", "Background Replacement - Stable Diffusion XL"], label='Select Model')
                    gr.Markdown("""
                                ### Instructions
                                - **Background Replacement - Stable Diffusion**:  
                                Required inputs: Input Image, Text Prompt, steps  
                                Specify the new background in the text box below.  
                                Example prompt: "dog sitting on the beach, sunny day, blue sky"
                                """)
                    prompt_background = gr.Textbox(label="Text Prompt: ")
                    steps_br = gr.Slider(minimum=0, maximum=200, label="Number of Steps", value=20, step=1)



        input_image.select(input_handler, inputs=[input_image], outputs=[coord_input, input_image])

        generate_button.click(
            fn=run_afm_app,
            inputs=[task_selector, input_image, mask_image, text_input, text_input_x, text_input_gsam, coord_input, ddim_steps, ddim_steps_pipe, 
                    inpaint_input_gsam, text_input_inpaint_pipe, text_input_restyling, blur, sharpen, prompt_outpaint, e_l, e_r, e_u, e_d, steps_outpaint,
                    prompt_background, steps_br, str_res, gs_res, np_res, steps_res, np_inpaint, steps_inpaint, prompt_txt2img, np_txt2img, gs_txt2img, 
                    steps_txt2img, steps_super, dilation_bool, dilation_value, steps_inp],
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

        reload_output_button.click(
            fn=reload_image_with_output,
            inputs=[output_image],
            outputs=[input_image]
        )

        reload_mask_button.click(
            fn=reload_mask,
            inputs=[output_image],
            outputs=[mask_image]
        )

        tab_task_selector_1.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_1], outputs=[task_selector])
        tab_task_selector_2.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_2], outputs=[task_selector])
        tab_task_selector_3.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_3], outputs=[task_selector])
        tab_task_selector_4.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_4], outputs=[task_selector])
        tab_task_selector_5.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_5], outputs=[task_selector])
        tab_task_selector_6.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_6], outputs=[task_selector])
        tab_task_selector_7.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_7], outputs=[task_selector])
        tab_task_selector_8.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_8], outputs=[task_selector])
        tab_task_selector_9.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_9], outputs=[task_selector])
        tab_task_selector_10.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_10], outputs=[task_selector])
        tab_task_selector_11.change(fn=update_task_selector, inputs=[task_selector, tab_task_selector_11], outputs=[task_selector])

    block.launch(share=True)