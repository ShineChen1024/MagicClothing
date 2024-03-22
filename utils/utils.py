import PIL
import numpy as np
import torch
import torch.nn.functional as F
import urllib.parse
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from huggingface_hub import (
    hf_hub_download,
    try_to_load_from_cache,
)
from functools import lru_cache
from garment_adapter import config



def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def prepare_image(image, height, width):
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Image as float32
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


def prepare_mask(image, height, width):
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 1, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.NEAREST) for i in image]
            image = [np.array(i.convert("L"))[..., None] for i in image]
            image = np.stack(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.stack([i[..., None] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 255.
        image[image > 0.5] = 1
        image[image <= 0.5] = 0

    return image


def extract_huggingface_repo_commit_file_from_url(url):
    parsed_url = urllib.parse.urlparse(url)
    path_components = parsed_url.path.strip("/").split("/")

    repo = "/".join(path_components[0:2])
    assert path_components[2] == "resolve"
    commit_hash = path_components[3]
    filepath = "/".join(path_components[4:])

    return repo, commit_hash, filepath

@lru_cache(maxsize=1)
def default_device():
    """determine whether to use cuda, mps, or cpu"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

def find_weights_path(weights_location: str) -> str:
    """
    Returns the local path to the weights file.

    Input can be a path, url, or predefined shortcut like "512" or "768"

    Downloads the weights file if not already cached locally.
    """

    # handle shortcuts
    weights_location = config.weight_shortcuts.get(weights_location, weights_location)

    if not weights_location.startswith("http"):
        return weights_location

    if "huggingface" not in weights_location:
        raise ValueError("URL must be from huggingface.co")

    repo, commit_hash, filepath = extract_huggingface_repo_commit_file_from_url(weights_location)
    dest_path = try_to_load_from_cache(
        repo_id=repo, revision=commit_hash, filename=filepath
    )
    if not dest_path:
        dest_path = hf_hub_download(repo_id=repo, revision=commit_hash, filename=filepath)
    return dest_path


def load_magic_clothing_model(
    model_weights=config.magic_clothing_diffusion_weights_default_url,
    enable_cloth_guidance=True,
    pipe_path="SG161222/Realistic_Vision_V4.0_noVAE",
    device=None,
):
    from garment_adapter.garment_diffusion import ClothAdapter
    from pipelines.OmsDiffusionPipeline import OmsDiffusionPipeline

    device = device or default_device()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
    if enable_cloth_guidance:
        pipe = OmsDiffusionPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    model_path = find_weights_path(model_weights)
    full_net = ClothAdapter(pipe, model_path, device, enable_cloth_guidance)
    return full_net


def load_magic_clothing_model_sd_inpainting(
    model_weights=config.magic_clothing_diffusion_weights_default_url,
    pipe_path="runwayml/stable-diffusion-inpainting",
    device=None,
):
    from garment_adapter.garment_diffusion import ClothAdapter
    from pipelines.OmsDiffusionInpaintPipeline import OmsDiffusionInpaintPipeline

    device = device or default_device()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
    pipe = OmsDiffusionInpaintPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    model_path = find_weights_path(model_weights)
    full_net = ClothAdapter(pipe, model_path, device, False)

    return full_net