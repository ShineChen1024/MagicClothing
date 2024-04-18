import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from humanparsing.run_parsing import Parsing
import argparse
import numpy as np
from PIL import ImageFilter
from tqdm import tqdm


class Matcher:
    def __init__(self, imgs, ft):
        self.ft = ft
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = 512

    def find_relations(self, x, y):
        num_channel = self.ft.size(1)
        with torch.no_grad():
            x, y = int(x), int(y)
            src_ft = self.ft[0].unsqueeze(0)
            src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
            src_vec = src_ft[0, :, y, x].view(1, num_channel)
            trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft[1:])
            trg_vec = trg_ft.view(self.num_imgs - 1, num_channel, -1)
            src_vec = F.normalize(src_vec)
            trg_vec = F.normalize(trg_vec)
            cos_map = torch.matmul(src_vec, trg_vec).view(self.num_imgs - 1, self.img_size, self.img_size).cpu().numpy()
            for i in range(1, self.num_imgs):
                max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)

        return max_yx[1].item(), max_yx[0].item()

def pad_to_patch_size(img, patch_size):
    width, height = img.size
    new_width = width + patch_size // 2 * 2
    new_height = height + patch_size
    result = Image.new(img.mode, (new_width, new_height), (0, 0, 0))
    result.paste(img, (patch_size // 2, patch_size // 2))
    return result

def get_relations(src_img, tgt_img, cloth_mask_filtered, src_points, lpips, patch_size):
    img_list = [src_img, tgt_img]
    src_img_tensor = (PILToTensor()(src_img) / 255.0 - 0.5) * 2
    tgt_img_tensor = (PILToTensor()(tgt_img) / 255.0 - 0.5) * 2
    dift = SDFeaturizer()
    ft = [dift.forward(src_img_tensor, prompt='a photo of an upper-garment', ensemble_size=8, t=41, up_ft_index=2),
         dift.forward(tgt_img_tensor, prompt='a photo of an upper-garment', ensemble_size=8, t=41, up_ft_index=2)]
    ft = torch.cat(ft, dim=0)
    matcher = Matcher(img_list, ft)
    img1 = pad_to_patch_size(src_img, patch_size)
    img2 = pad_to_patch_size(tgt_img, patch_size)
    for i in tqdm(range(len(src_points))):
        tgt_index = matcher.find_relations(src_points[i][1], src_points[i][0])
        if cloth_mask_filtered[tgt_index[1], tgt_index[0]] == 0:
            lpips.update(torch.ones(1, 3, patch_size, patch_size).cuda(), torch.zeros(1, 3, patch_size, patch_size).cuda())
        else:
            patch1 = img1.crop((src_points[i][1], src_points[i][0], src_points[i][1] + patch_size, src_points[i][0] + patch_size))
            patch2 = img2.crop((tgt_index[0], tgt_index[1], tgt_index[0] + patch_size, tgt_index[1] + patch_size))
            lpips.update(transforms.ToTensor()(patch1).unsqueeze(0).cuda(), transforms.ToTensor()(patch2).unsqueeze(0).cuda())
    return lpips



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the metrics for the generated images")
    parser.add_argument('--image_path', type=str, help='generated image path')
    parser.add_argument('--cloth_path', type=str, help='input cloth path')
    parser.add_argument('--cloth_mask_path', type=str, help='input cloth mask path')
    parser.add_argument("--point_distance", type=int, default=40, help="chosen point density, higher indicates sparser")
    parser.add_argument("--patch_size", type=int, default=33, help="patch size for matched points")
    args = parser.parse_args()
    torch.manual_seed(0)
    parsing_model = Parsing()
    model_img = Image.open(args.image_path).resize((512, 512)).convert('RGB')
    cloth_img = Image.open(args.cloth_path).resize((512, 512)).convert('RGB')
    cloth_mask = Image.open(args.cloth_mask_path).resize((512, 512)).convert('L')
    cloth_mask = (np.array(cloth_mask) > 127).astype(np.uint8)
    cloth_masked = Image.fromarray(((cloth_mask == 1)[..., None] * np.array(cloth_img)).astype(np.uint8))
    model_cloth_mask = parsing_model(model_img)
    model_cloth_masked = Image.fromarray(((np.array(model_cloth_mask) == 255)[..., None] * np.array(model_img)).astype(np.uint8))
    model_cloth_mask_filtered = np.array(Image.fromarray(model_cloth_mask).filter(ImageFilter.MaxFilter(17)).convert('L'))
    list_indexes = np.argwhere((cloth_mask == 1))
    chosen_indexes = [(list_index[0], list_index[1]) for list_index in list_indexes if ((list_index[0] % args.point_distance == 0) and (list_index[1] % args.point_distance == 0))]
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to('cuda')
    mp_lpips = get_relations(cloth_masked, model_cloth_masked, model_cloth_mask_filtered, chosen_indexes, lpips, args.patch_size)
    print("MP-LPIPS value is %f" % mp_lpips.compute().item())
