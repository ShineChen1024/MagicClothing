import os

# tells pytorch to allow MPS usage (for Mac M1 compatibility)
os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from controlnet_aux import OpenposeDetector
import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL, ControlNetModel
from diffusers.pipelines import StableDiffusionControlNetPipeline
import gradio as gr
import argparse
import cv2

from garment_adapter import config
from pipelines.OmsDiffusionControlNetPipeline import OmsDiffusionControlNetPipeline
from utils.utils import find_weights_path, default_device

parser = argparse.ArgumentParser(description='oms diffusion')
parser.add_argument('--model_weights', type=str, default=config.magic_clothing_diffusion_weights_default_url)
parser.add_argument('--pipe_path', type=str, default="SG161222/Realistic_Vision_V4.0_noVAE")
parser.add_argument('--enable_cloth_guidance', action="store_true")
parser.add_argument('--faceid_version', type=str, default="FaceIDPlus", choices=['FaceID', 'FaceIDPlus', 'FaceIDPlusV2'])

args = parser.parse_args()

openpose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(default_device())
control_net_openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
if args.enable_cloth_guidance:
    pipe = OmsDiffusionControlNetPipeline.from_pretrained(args.pipe_path, vae=vae, controlnet=control_net_openpose, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(args.pipe_path, vae=vae, controlnet=control_net_openpose, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
model_path = find_weights_path(args.model_weights)

if args.faceid_version == "FaceID":
    ip_lora = find_weights_path(config.faceid_lora_weights)
    ip_ckpt = find_weights_path(config.faceid_ckpt_weights)
    pipe.load_lora_weights(ip_lora)

    pipe.fuse_lora()
    from garment_adapter.garment_ipadapter_faceid import IPAdapterFaceID

    ip_model = IPAdapterFaceID(pipe, model_path, ip_ckpt, args.enable_cloth_guidance)
else:
    if args.faceid_version == "FaceIDPlus":
        ip_lora = find_weights_path(config.faceid_plus_lora_weights)
        ip_ckpt = find_weights_path(config.faceid_plus_ckpt_weights)
        v2 = False
    else:
        ip_lora = find_weights_path(config.faceid_plusv2_lora_weights)
        ip_ckpt = find_weights_path(config.faceid_plusv2_ckpt_weights)
        v2 = True

    pipe.load_lora_weights(ip_lora)
    pipe.fuse_lora()

    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    from garment_adapter.garment_ipadapter_faceid import IPAdapterFaceIDPlus as IPAdapterFaceID

    ip_model = IPAdapterFaceID(pipe, model_path, image_encoder_path, ip_ckpt, args.enable_cloth_guidance)


def process(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, width, height, sample_steps, scale, cloth_guidance_scale, seed, pose_image):
    if args.faceid_version == "FaceID":
        result = ip_model.generate(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, cloth_guidance_scale, sample_steps, height, width, image=pose_image)
    else:
        result = ip_model.generate(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, cloth_guidance_scale, sample_steps, height, width, shortcut=v2, image=pose_image)
    if result is None:
        raise gr.Error("人脸检测异常，尝试其他肖像")
    else:
        images, cloth_mask_image = result
    return images, cloth_mask_image


def get_pose(image):
    openpose_image = openpose_model(image)
    return openpose_image


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
            pose_image = gr.Image(label="pose Image", type="pil")
            pose_button = gr.Button(value="get pose")
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
            cloth_seg_image = gr.Image(label="cloth mask", type="pil", width=192, height=256)

    ips = [cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, width, height, sample_steps, guidance_scale, cloth_guidance_scale, seed, pose_image]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, cloth_seg_image])
    pose_button.click(fn=get_pose, inputs=pose_image, outputs=pose_image)

block.launch(server_name="0.0.0.0", server_port=7860)
