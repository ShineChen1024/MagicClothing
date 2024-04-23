# Magic Clothing Inpainting guidance

We upload ***gradio_cloth_inpainting.py*** in response to the enthusiasm for virtual try-on.

1. we suggest you to upload the cloth mask or use other strong model to seg out the cloth from background if the cloth_segm.pth generates wrong result.
2. This gradio demo need you to mask out the original cloth yourself, but you can utilize the algorithm generates mask automatically in our [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) 
3. we support different cloth type, e.g. jackets, coat, underwear which are not in VITON-HD, sometimes input the garment type in the prompt may help you!

Anyway, have fun!
