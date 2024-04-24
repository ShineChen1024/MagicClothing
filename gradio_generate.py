import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
import gradio as gr
import argparse

from garment_adapter.garment_diffusion import ClothAdapter
from pipelines.OmsDiffusionPipeline import OmsDiffusionPipeline
#from pipelines.OmsIndependentDiffusionPipeline import OmsIndependentDiffusionPipeline

parser = argparse.ArgumentParser(description='oms diffusion')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--enable_cloth_guidance', action="store_true")
parser.add_argument('--use_independent_condition', action="store_true")
parser.add_argument('--pipe_path', type=str, default="SG161222/Realistic_Vision_V4.0_noVAE") #stablediffusionapi/counterfeit-v30 SG161222/Realistic_Vision_V4.0_noVAE

args = parser.parse_args()

device = "cuda"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
if args.enable_cloth_guidance:
    if args.use_independent_condition:
        pipe = OmsIndependentDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, torch_dtype=torch.float16, safety_checker=None)
    else:
        pipe = OmsDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, torch_dtype=torch.float16, safety_checker=None)

else:
    pipe = StableDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
full_net = ClothAdapter(pipe, args.model_path, device, args.enable_cloth_guidance, args.use_independent_condition)


def process(cloth_image, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, width, height, sample_steps, scale, cloth_guidance_scale, seed):
    images, cloth_mask_image = full_net.generate(cloth_image, cloth_mask_image, prompt, a_prompt, num_samples, n_prompt, seed, scale, cloth_guidance_scale, sample_steps, height, width)
    return images, cloth_mask_image


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("##You can enlarge image resolution to get better face, but the cloth maybe lose control, we will release high-resolution checkpoint soon##")
    with gr.Row():
        with gr.Column():
            cloth_image = gr.Image(label="cloth Image", type="pil")
            cloth_mask_image = gr.Image(label="cloth mask Image, if not support, will be produced by inner segment algorithm", type="pil")
            prompt = gr.Textbox(label="Prompt", value='a photography of a model')
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                height = gr.Slider(label="Height", minimum=256, maximum=1024, value=768, step=64)
                width = gr.Slider(label="Width", minimum=192, maximum=768, value=576, step=64)
                sample_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10., value=5. if args.enable_cloth_guidance else 2.5, step=0.1)
                cloth_guidance_scale = gr.Slider(label="Cloth guidance Scale", minimum=1, maximum=10., value=2.5, step=0.1, visible=args.enable_cloth_guidance)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1234)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, high quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='bare, monochrome, lowres, bad anatomy, worst quality, low quality')

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
            cloth_seg_image = gr.Image(label="cloth mask", type="pil", width=192, height=256)

    ips = [cloth_image, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, width, height, sample_steps, guidance_scale, cloth_guidance_scale, seed]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, cloth_seg_image])

block.launch(server_name="0.0.0.0", server_port=7860)
