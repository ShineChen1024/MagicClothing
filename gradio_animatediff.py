import os.path
import pdb

import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL, DDIMScheduler, MotionAdapter, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler,StableVideoDiffusionPipeline
from diffusers.pipelines import AnimateDiffPipeline
from PIL import Image
import argparse
from diffusers.utils import export_to_gif
from garment_adapter.garment_diffusion import ClothAdapter_AnimateDiff
from pipelines.OmsAnimateDiffusionPipeline import OmsAnimateDiffusionPipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='oms diffusion')
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--pipe_path', type=str, default="SG161222/Realistic_Vision_V4.0_noVAE")
    parser.add_argument('--output_path', type=str, default="./output_img")

    args = parser.parse_args()

    device = "cuda"
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cloth_image = Image.open(args.cloth_path).convert("RGB")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

    pipe = OmsAnimateDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, motion_adapter=adapter, torch_dtype=torch.float16)
    # pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

    full_net = ClothAdapter_AnimateDiff(pipe, args.pipe_path, args.model_path, device)
    frames, cloth_mask_image = full_net.generate(cloth_image, num_images_per_prompt=1, seed=6896868)
    export_to_gif(frames[0], "animation0.gif")
