import torch.nn.functional as F
import numpy as np
import PIL
import torch


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
