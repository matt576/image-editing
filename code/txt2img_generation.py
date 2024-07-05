from diffusers import AutoPipelineForText2Image
import torch

def txt2img_gradio(input_image, task_selector, prompt, gs, steps, negative_prompt):
    print(task_selector)

    gs = float(gs)
    steps = int(steps)
    
    if task_selector == "Stable Diffusion v1.5 Txt2Img":

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        generator = torch.Generator("cuda").manual_seed(31)
        image = pipeline(prompt=prompt, 
                        generator=generator,
                        height=512,
                        width=512,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=gs
                        ).images[0]

    
    elif task_selector == "Stable Diffusion XL Txt2Img":

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        generator = torch.Generator("cuda").manual_seed(31)
        image = pipeline(prompt=prompt, 
                        generator=generator,
                        height=1024,
                        width=1024,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=gs
                        ).images[0]

    elif task_selector == "Kandinsky v2.2 Txt2Img":

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ).to("cuda")
        generator = torch.Generator("cuda").manual_seed(31)
        image = pipeline(prompt=prompt, 
                        generator=generator,
                        height=512,
                        width=512,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=gs
                        ).images[0]
    else:
        print("Please pick a valid model: SD, SDXL, Kandinsky")


    image.save("outputs/txt2img/generated_input.png", format="PNG", optimize=True, compress_level=6)

    return image



if __name__ == "__main__":

    # Pick model:
    # txt2img_model = "SD"
    # txt2img_model = "SDXL"
    txt2img_model = "Kandinsky"

    print(txt2img_model)

    prompt = "Batman on top of skyscraper overlooking gotham city, sunny, blue sky, photorealistic, high quality, 8k"
    negative_prompt = "ugly, bad quality, poor details, deformed"
    steps = 50
    gs = 3.5

    if txt2img_model == "SD":

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        generator = torch.Generator("cuda").manual_seed(31)
        image = pipeline(prompt=prompt, 
                        generator=generator,
                        height=512,
                        width=512,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=gs
                        ).images[0]

    
    elif txt2img_model == "SDXL":

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        generator = torch.Generator("cuda").manual_seed(31)
        image = pipeline(prompt=prompt, 
                        generator=generator,
                        height=1024,
                        width=1024,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=gs
                        ).images[0]

    elif txt2img_model == "Kandinsky":

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ).to("cuda")
        generator = torch.Generator("cuda").manual_seed(31)
        image = pipeline(prompt=prompt, 
                        generator=generator,
                        height=512,
                        width=512,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=gs
                        ).images[0]
    else:
        print("Please pick a valid model: SD, SDXL, Kandinsky")


    image.save("outputs/txt2img/tx2img_output.png", format="PNG", optimize=True, compress_level=6)