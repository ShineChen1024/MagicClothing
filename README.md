# Magic Clothing
This repository is the official implementation of Magic Clothing

Magic Clothing is a branch version of [OOTDiffusion](https://github.com/levihsu/OOTDiffusion), focusing on controllable garment-driven image synthesis

<!-- Please refer to our [previous paper](https://arxiv.org/abs/2403.01779) for more details -->

> **Magic Clothing: Controllable Garment-Driven Image Synthesis** [[arXiv paper](https://arxiv.org/abs/2404.09512)]<br>
> [Weifeng Chen](https://github.com/ShineChen1024)\*, [Tao Gu](https://github.com/T-Gu)\*, [Yuhao Xu](http://levihsu.github.io/), [Chengcai Chen](https://www.researchgate.net/profile/Chengcai-Chen)<br>
> \* Equal contribution<br>
> Xiao-i Research


## News
🔥 [2024/4/23] In response to the enthusiasm for cloth inpaiting task, see our [guidance](https://github.com/ShineChen1024/MagicClothing/blob/main/virtual_tryon_img/virtual_tryon_guidance.md)!

🔥 [2024/4/19] An 1024 version trained on both VTON-HD and DressCode for early access branch is avaliable now!

🔥 [2024/4/19] We support AnimateDiff now for generating GIF!

🔥 [2024/4/16] Our [paper](https://arxiv.org/abs/2404.09512) is available now!

🔥 [2024/3/8] We released the model weights trained on the 768 resolution. The strength of clothing and text prompts can be independently adjusted.

🤗 [Hugging Face link](https://huggingface.co/ShineChen1024/MagicClothing)

🔥 [2024/2/28] We support [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) with [ControlNet-Openpose](https://github.com/lllyasviel/ControlNet-v1-1-nightly)! A portrait and a reference pose image can be used as additional conditions.

Have fun with ***gradio_ipadapter_openpose.py***

🔥 [2024/2/23] We support [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) now! A portrait image can be used as an additional condition.

Have fun with ***gradio_ipadapter_faceid.py***

![demo](images/demo.png)&nbsp;
![workflow](images/workflow.png)&nbsp;

***Cloth Inpainting Demo***
<div align="left">
    <img src="virtual_tryon_img/a1.jpg" alt="图片1" width="10%">
    <img src="virtual_tryon_img/a2.png" alt="图片2" width="10%">
    <img src="virtual_tryon_img/a3.png" alt="图片3" width="10%">
    <img src="virtual_tryon_img/b1.jpg" alt="图片4" width="10%">
    <img src="virtual_tryon_img/b2.png" alt="图片5" width="10%">
    <img src="virtual_tryon_img/b3.png" alt="图片6" width="10%">
    <img src="virtual_tryon_img/c1.jpg" alt="图片7" width="10%">
    <img src="virtual_tryon_img/c2.png" alt="图片8" width="10%">
    <img src="virtual_tryon_img/c3.png" alt="图片9" width="10%">
</div>

***1024 version for upper-body lower-body and full-body clothes Demo***
<div align="left">
    <img src="images/a0.jpg" alt="图片1" width="15%">
    <img src="images/a1.png" alt="图片2" width="15%">
    <img src="images/b0.jpg" alt="图片3" width="15%">
    <img src="images/b1.png" alt="图片4" width="15%">
    <img src="images/c0.jpg" alt="图片5" width="15%">
    <img src="images/c1.png" alt="图片6" width="15%">
</div>

***AnimateDiff Demo*** 'a beautiful girl with a smile' 
<div align="center">
    <img src="valid_cloth/t1.png" width="15%">
    <img src="images/animatediff0.gif" alt="图片1" width="15%">
    <img src="valid_cloth/t6.png" width="15%">
    <img src="images/animatediff1.gif" alt="图片2" width="15%">
    <img src="valid_cloth/t7.jpg" width="13%">
    <img src="images/animatediff2.gif" alt="图片3" width="15%">
</div>

## Installation

1. Clone the repository

```sh
git clone https://github.com/ShineChen1024/MagicClothing.git
```

2. Create a conda environment and install the required packages

```sh
conda create -n magicloth python==3.10
conda activate magicloth
pip install torch==2.0.1 torchvision==0.15.2 numpy==1.25.1 diffusers==0.25.1 opencv-python==4.9.0.80  transformers==4.31.0 gradio==4.16.0 safetensors==0.3.1 controlnet-aux==0.0.6 accelerate==0.21.0
```

## Inference

1. Python demo

> 512 weights

```sh
python inference.py --cloth_path [your cloth path] --model_path [your model checkpoints path]
```

> 768 weights

```sh
python inference.py --cloth_path [your cloth path] --model_path [your model checkpoints path] --enable_cloth_guidance
```

2. Gradio demo

> 512 weights

```sh
python gradio_generate.py --model_path [your model checkpoints path] 
```

> 768 weights

```sh
python gradio_generate.py --model_path [your model checkpoints path] --enable_cloth_guidance
```

## Citation
```
@article{chen2024magic,
  title={Magic Clothing: Controllable Garment-Driven Image Synthesis},
  author={Chen, Weifeng and Gu, Tao and Xu, Yuhao and Chen, Chengcai},
  journal={arXiv preprint arXiv:2404.09512},
  year={2024}
}
```

## TODO List
- [x] Paper
- [x] Gradio demo
- [x] Inference code
- [x] Model weights
- [ ] Training code
