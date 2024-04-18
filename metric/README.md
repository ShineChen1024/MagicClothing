# Metric MP-LPIPS Calculation

## 1. DIFT
The calculation of MP-LPIPS is based on diffusion features (DIFT), Please check https://github.com/Tsingularity/dift for setting up the environment.

## 2. Human parse
Install `onnxruntime==1.16.2`.
We modify the code from https://github.com/levihsu/OOTDiffusion for [humanparsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) to accomplish garment segmentation.
Please download the ONNX checkpoint at [Hugging Face Link]https://huggingface.co/levihsu/OOTDiffusion/blob/main/checkpoints/humanparsing/parsing_atr.onnx and place it into ***checkpoints/humanparsing*** folder.

## 3. MP_LPIPS calculation
For quick test, you can get cloth and cloth mask from [VITON-HD](https://github.com/shadow2496/VITON-HD) and the image generated using our Magic Clothing.
```sh
python metric_MP_LPIPS.py --image_path 'test image path' --cloth_path 'test cloth path' --cloth_mask_path 'test mask path'
```