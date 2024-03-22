import argparse
import gradio as gr

from garment_adapter import config
from utils.utils import load_magic_clothing_model_sd_inpainting

parser = argparse.ArgumentParser(description='oms diffusion')
parser.add_argument('--model_weights', type=str, default=config.magic_clothing_diffusion_weights_default_url)
parser.add_argument('--pipe_path', type=str, default="runwayml/stable-diffusion-inpainting")

args = parser.parse_args()

full_net = load_magic_clothing_model_sd_inpainting(model_weights=args.model_weights, pipe_path=args.pipe_path)



def process(person_image, person_mask, cloth_image, cloth_mask_image, num_samples, width, height, sample_steps, cloth_guidance_scale, seed):
    # person_image = person_image_mask['background'].convert("RGB")
    # person_mask = person_image_mask['layers'][0].split()[-1]

    images, cloth_mask_image = full_net.generate_inpainting(cloth_image, cloth_mask_image, num_samples, seed, cloth_guidance_scale, sample_steps, height, width, image=person_image, mask_image=person_mask)
    return images, cloth_mask_image


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("##You can enlarge image resolution to get better face, but the cloth maybe lose control, we will release high-resolution checkpoint soon##")
    with gr.Row():
        with gr.Column():
            cloth_image = gr.Image(label="cloth Image", type="pil")
            cloth_mask_image = gr.Image(label="cloth mask Image, if not support, will be produced by inner segment algorithm", type="pil")
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                height = gr.Slider(label="Height", minimum=256, maximum=1024, value=1024, step=64)
                width = gr.Slider(label="Width", minimum=192, maximum=768, value=768, step=64)
                sample_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                cloth_guidance_scale = gr.Slider(label="Cloth guidance Scale", minimum=1, maximum=10., value=2.5, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1234)
        with gr.Column():
            person_image = gr.Image(label="person Image", type="pil")
            person_mask = gr.Image(label="person mask", type="pil")
            # person_image_mask = gr.ImageMask(label="person Image", type="pil")
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
            cloth_seg_image = gr.Image(label="cloth mask", type="pil", width=192, height=256)

    ips = [person_image, person_mask, cloth_image, cloth_mask_image, num_samples, width, height, sample_steps, cloth_guidance_scale, seed]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, cloth_seg_image])

block.launch(server_name="0.0.0.0", server_port=7860)
