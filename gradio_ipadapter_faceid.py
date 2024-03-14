import os

from PIL import Image
import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
import gradio as gr
import argparse
import cv2

from pipelines.OmsDiffusionPipeline import OmsDiffusionPipeline

parser = argparse.ArgumentParser(description='oms diffusion')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--pipe_path', type=str, default="SG161222/Realistic_Vision_V4.0_noVAE")
parser.add_argument('--enable_cloth_guidance', action="store_true")
parser.add_argument('--faceid_version', type=str, default="FaceIDPlusV2", choices=['FaceID', 'FaceIDPlus', 'FaceIDPlusV2'])

args = parser.parse_args()

device = "cuda"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
if args.enable_cloth_guidance:
    pipe = OmsDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

if args.faceid_version == "FaceID":
    ip_lora = "./checkpoints/ipadapter_faceid/ip-adapter-faceid_sd15_lora.safetensors"
    ip_ckpt = "./checkpoints/ipadapter_faceid/ip-adapter-faceid_sd15.bin"
    pipe.load_lora_weights(ip_lora)
    pipe.fuse_lora()
    from garment_adapter.garment_ipadapter_faceid import IPAdapterFaceID

    ip_model = IPAdapterFaceID(pipe, args.model_path, ip_ckpt, device, args.enable_cloth_guidance)
else:
    if args.faceid_version == "FaceIDPlus":
        ip_ckpt = "./checkpoints/ipadapter_faceid/ip-adapter-faceid-plus_sd15.bin"
        ip_lora = "./checkpoints/ipadapter_faceid/ip-adapter-faceid-plus_sd15_lora.safetensors"
        v2 = False
    else:
        ip_ckpt = "./checkpoints/ipadapter_faceid/ip-adapter-faceid-plusv2_sd15.bin"
        ip_lora = "./checkpoints/ipadapter_faceid/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
        v2 = True

    pipe.load_lora_weights(ip_lora)
    pipe.fuse_lora()
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    from garment_adapter.garment_ipadapter_faceid import IPAdapterFaceIDPlus as IPAdapterFaceID

    ip_model = IPAdapterFaceID(pipe, args.model_path, image_encoder_path, ip_ckpt, device, args.enable_cloth_guidance)


def process(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, width, height, sample_steps, scale, cloth_guidance_scale, seed):
    if args.faceid_version == "FaceID":
        result = ip_model.generate(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, cloth_guidance_scale, sample_steps, height, width)
    else:
        result = ip_model.generate(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, cloth_guidance_scale, sample_steps, height, width, shortcut=v2)
    if result is None:
        raise gr.Error("人脸检测异常，尝试其他肖像")
    else:
        images, cloth_mask_image = result
    return images, cloth_mask_image


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("##You can enlarge image resolution to get better face, but the cloth maybe lose control, we will release high-resolution checkpoint soon##")
    with gr.Row():
        with gr.Column():
            face_img = gr.Image(label="face Image", type="pil")
            cloth_image = gr.Image(label="cloth Image", type="pil")
            cloth_mask_image = gr.Image(label="cloth mask Image, if not support, will be produced by inner segment algorithm", type="pil")
            prompt = gr.Textbox(label="Prompt", value='a photography')
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                height = gr.Slider(label="Height", minimum=256, maximum=1024, value=768, step=64)
                width = gr.Slider(label="Width", minimum=192, maximum=768, value=576, step=64)
                sample_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10., value=3. if args.enable_cloth_guidance else 2.5, step=0.1)
                cloth_guidance_scale = gr.Slider(label="Cloth guidance Scale", minimum=1, maximum=10., value=3., step=0.1, visible=args.enable_cloth_guidance)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1234)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, high quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='bare, monochrome, lowres, bad anatomy, worst quality, low quality')

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
            cloth_seg_image = gr.Image(label="cloth mask", type="pil", width=192, height=256)

    ips = [cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, width, height, sample_steps, guidance_scale, cloth_guidance_scale, seed]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, cloth_seg_image])

block.launch(server_name="0.0.0.0", server_port=7860)
