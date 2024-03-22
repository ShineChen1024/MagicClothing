import argparse
import os.path
from PIL import Image

from garment_adapter import config
from utils.utils import load_magic_clothing_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='oms diffusion')
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--model_weights', type=str, default=config.magic_clothing_diffusion_weights_default_url)
    parser.add_argument('--enable_cloth_guidance', action="store_true")
    parser.add_argument('--pipe_path', type=str, default="SG161222/Realistic_Vision_V4.0_noVAE")
    parser.add_argument('--output_path', type=str, default="./output_img")

    args = parser.parse_args()

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    cloth_image = Image.open(args.cloth_path).convert("RGB")

    full_net = load_magic_clothing_model(args.model_weights, args.enable_cloth_guidance, args.pipe_path)

    images = full_net.generate(cloth_image)
    for i, image in enumerate(images[0]):
        image.save(os.path.join(output_path, f"out_{i:03d}.png"))
