import torch
from utils.utils import is_torch2_available

from .garment_ipadapter_faceid import IPAdapterFaceIDPlus, IPAdapterFaceID

USE_DAFAULT_ATTN = False  # should be True for visualization_attnmap
if is_torch2_available() and (not USE_DAFAULT_ATTN):
    from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
    from .attention_processor import StableREFAttnProcessor2_0 as StableREFAttnProcessor
else:
    from .attention_processor import AttnProcessor, IPAttnProcessor, REFAttnProcessor


class StableIPAdapterFaceID(IPAdapterFaceID):
    def __init__(self, sd_pipe, ref_path, self_ip_path, image_encoder_path, ip_ckpt, device, enable_cloth_guidance, num_tokens=4, torch_dtype=torch.float16, set_seg_model=True):
        super().__init__(sd_pipe, ref_path, image_encoder_path, ip_ckpt, device, enable_cloth_guidance, num_tokens, torch_dtype, set_seg_model)
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(torch.load(self_ip_path, map_location="cpu"), strict=False)
        ip_layers.to(self.device)

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = StableREFAttnProcessor(hidden_size=hidden_size, cross_attention_dim=hidden_size, name=name)
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.torch_dtype)
        unet.set_attn_processor(attn_procs)


class StableIPAdapterFaceIDPlus(IPAdapterFaceIDPlus):
    def __init__(self, sd_pipe, ref_path, self_ip_path, image_encoder_path, ip_ckpt, device, enable_cloth_guidance, num_tokens=4, torch_dtype=torch.float16, set_seg_model=True):
        super().__init__(sd_pipe, ref_path, image_encoder_path, ip_ckpt, device, enable_cloth_guidance, num_tokens, torch_dtype, set_seg_model)
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(torch.load(self_ip_path, map_location="cpu"), strict=False)
        ip_layers.to(self.device)

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = StableREFAttnProcessor(hidden_size=hidden_size, cross_attention_dim=hidden_size, name=name)
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.torch_dtype)
        unet.set_attn_processor(attn_procs)
