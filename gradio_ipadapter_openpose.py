import os

from controlnet_aux import OpenposeDetector
import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL, ControlNetModel
from diffusers.pipelines import StableDiffusionControlNetPipeline
import gradio as gr
import argparse
import cv2

parser = argparse.ArgumentParser(description='oms diffusion')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--pipe_path', type=str, default="SG161222/Realistic_Vision_V4.0_noVAE")
parser.add_argument('--faceid_version', type=str, default="FaceIDPlus", choices=['FaceID', 'FaceIDPlus', 'FaceIDPlusV2'])

args = parser.parse_args()

device = "cuda"

openpose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device)
control_net_openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(args.pipe_path, vae=vae, controlnet=control_net_openpose, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

if args.faceid_version == "FaceID":
    ip_lora = "./checkpoints/ipadapter_faceid/ip-adapter-faceid_sd15_lora.safetensors"
    ip_ckpt = "./checkpoints/ipadapter_faceid/ip-adapter-faceid_sd15.bin"
    pipe.load_lora_weights(ip_lora)

    pipe.fuse_lora()
    from garment_adapter.garment_ipadapter_faceid import IPAdapterFaceID

    ip_model = IPAdapterFaceID(pipe, args.model_path, ip_ckpt, device)
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

    ip_model = IPAdapterFaceID(pipe, args.model_path, image_encoder_path, ip_ckpt, device)


def process(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, width, height, sample_steps, scale, seed, pose_image):
    if args.faceid_version == "FaceID":
        result = ip_model.generate(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, sample_steps, height, width, image=pose_image)
    else:
        result = ip_model.generate(cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, sample_steps, height, width, shortcut=v2, image=pose_image)
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
            prompt = gr.Textbox(label="Prompt", value='a photography of a model')
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=4, step=1)
                height = gr.Slider(label="Height", minimum=256, maximum=768, value=512, step=64)
                width = gr.Slider(label="Width", minimum=192, maximum=576, value=384, step=64)
                sample_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10., value=2.5, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1234)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, high quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='bare, monochrome, lowres, bad anatomy, worst quality, low quality')
        with gr.Column():
            pose_image = gr.Image(label="pose Image", type="pil")
            pose_button = gr.Button(value="get pose")
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
            cloth_seg_image = gr.Image(label="cloth mask", type="pil", width=192, height=256)

    ips = [cloth_image, face_img, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, width, height, sample_steps, scale, seed, pose_image]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, cloth_seg_image])
    pose_button.click(fn=get_pose, inputs=pose_image, outputs=pose_image)

block.launch(server_name="0.0.0.0", server_port=7860)
