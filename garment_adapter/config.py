
_hf_base_url = "https://huggingface.co/ShineChen1024/oms-diffusion/resolve/3f90dd01926f0053e61c9c80fa82d3629c3a54cb"


magic_clothing_diffusion_weights_512_url = f"{_hf_base_url}/oms_diffusion_100000.safetensors"
magic_clothing_diffusion_weights_768_url = f"{_hf_base_url}/oms_diffusion_768_200000.safetensors"
magic_clothing_diffusion_weights_default_url = magic_clothing_diffusion_weights_768_url
cloth_segmentation_weights_url = f"{_hf_base_url}/cloth_segm.pth"

_ip_adapter_face_base_url = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/f6494dd606604d7778561f570fdc4b56d39b1fbb"

faceid_lora_weights = f"{_ip_adapter_face_base_url}/ip-adapter-faceid_sd15_lora.safetensors"
faceid_ckpt_weights = f"{_ip_adapter_face_base_url}/ip-adapter-faceid_sd15.bin"

faceid_plus_lora_weights = f"{_ip_adapter_face_base_url}/ip-adapter-faceid-plus_sd15_lora.safetensors"
faceid_plus_ckpt_weights = f"{_ip_adapter_face_base_url}/ip-adapter-faceid-plus_sd15.bin"

faceid_plusv2_lora_weights = f"{_ip_adapter_face_base_url}/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
faceid_plusv2_ckpt_weights = f"{_ip_adapter_face_base_url}/ip-adapter-faceid-plusv2_sd15.bin"


weight_shortcuts = {
    "512": magic_clothing_diffusion_weights_512_url,
    "768": magic_clothing_diffusion_weights_768_url,
    "cloth_segm": cloth_segmentation_weights_url,
}
