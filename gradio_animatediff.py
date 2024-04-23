import os.path
import pdb

import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL, DDIMScheduler, MotionAdapter, \
    EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, StableVideoDiffusionPipeline
from diffusers.pipelines import AnimateDiffPipeline
from PIL import Image
import argparse
from diffusers.utils import export_to_gif
from garment_adapter.garment_diffusion import ClothAdapter_AnimateDiff
from pipelines.OmsAnimateDiffusionPipeline import OmsAnimateDiffusionPipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='oms diffusion')
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--pipe_path', type=str, default="/home/aigc/HuggingFace_models/realisticvision_v4")
    parser.add_argument('--output_path', type=str, default="./output_img")

    args = parser.parse_args()

    device = "cuda"
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cloth_image = Image.open(args.cloth_path).convert("RGB")

    vae = AutoencoderKL.from_pretrained("/home/aigc/HuggingFace_models/sd-vae-ft-mse").to(dtype=torch.float16)
    adapter = MotionAdapter.from_pretrained("/home/aigc/HuggingFace_models/animatediff", torch_dtype=torch.float16)

    pipe = OmsAnimateDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, motion_adapter=adapter,
                                                       torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(
        args.pipe_path,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler
    # pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    garment_extractor_path = os.path.join(args.ckpt_dir, "garment_extractor.safetensors")
    garment_ip_layer_path = os.path.join(args.ckpt_dir, "ip_layer.pth")
    full_net = ClothAdapter_AnimateDiff(pipe, args.pipe_path, garment_extractor_path, garment_ip_layer_path, device)
    frames, cloth_mask_image = full_net.generate(cloth_image, num_images_per_prompt=1, seed=756464)
    export_to_gif(frames[0], os.path.join(output_path, "animation0.gif"))
