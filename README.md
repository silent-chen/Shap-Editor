# SHAP-EDITOR: Instruction-guided Latent 3D Editing in Seconds (CVPR 2024)

Minghao Chen, Junyu Xie, Iro Laina, Andrea Vedaldi

[Paper](https://arxiv.org/abs/2312.09246) | [Webpage](https://silent-chen.github.io/Shap-Editor/) | [Demo](https://huggingface.co/spaces/silentchen/Shap_Editor_demo)

<div align="center">
    <img width="100%" alt="teaser" src="https://github.com/silent-chen/Shap-Editor/blob/gh-page/resources/teaser_v4.png?raw=true"/>
</div>
We present a method, named Shap-Editor, aiming at fast 3D editing. We propose to learn a universal editing function that can be applied to different objects within one second.


## News:
- Jun 18, 2024 :fire: :fire: : The inference code is available. Try to modify your own 3D object in seconds!
- Feb 26, 2024 :fire:: Shap-Editor is accepted in CVPR 2024!!!
- Dec 15, 2023: Demo is available [here](https://huggingface.co/spaces/silentchen/Shap_Editor_demo) on Hugging Face.

## Environment Setup

To set up the environment you can easily run the following command:
```buildoutcfg
conda create -n Shap-Editor python=3.9
conda activate Shap-Editor
pip install -r requirements.txt
```
## Pre-trained Models
We provide some pre-trained editing models for playing around.

| prompt | Link                                                                                                 | Model Type |
| -------- |------------------------------------------------------------------------------------------------------|--------|
| "Add a santa hat to it" | [Link](https://huggingface.co/silentchen/Shap_Editor/resolve/main/single/santa_hat.pt?download=true) | Single |
| "Make it look like made of gold" | [Link](https://huggingface.co/silentchen/Shap_Editor/resolve/main/single/golden.pt?download=true)    | Single |
| "Make the color of it look like rainbow" | [Link](https://huggingface.co/silentchen/Shap_Editor/resolve/main/single/rainbow.pt?download=true)   | Single |
| "Make it in cyberpunk style" | [Link](https://huggingface.co/silentchen/Shap_Editor/resolve/main/single/cyber.pt?download=true)     | Single |
| "Make it wooden" | [Link](https://huggingface.co/silentchen/Shap_Editor/resolve/main/single/wooden.pt?download=true)    | Single |
| "Make it look like make of lego" | [Link](https://huggingface.co/silentchen/Shap_Editor/resolve/main/single/lego.pt?download=true)      | Single |



## Demo
We provide demo in following ways:
1. You can try our [Gradio demo](https://huggingface.co/spaces/silentchen/Shap_Editor_demo) on Hugging Face. (Thank Hugging Face for providing free GPU support!)
2. You can also run the demo locally using [`python app.py`].

## Inference
### Step 1: Get the latent
To edit a 3D assets, we need its corresponding latent. In this work, we consider two main sources of latents:
1. Latents generate by Shap-E diffusion models. You can generate your own 3D assets with latent by use:
```buildoutcfg
python generate_3d.py --prompt 'a shark' -output_dir PATH_TO_OUTPUT_DIR
```
2. Encoded Latents from real 3D assets. You can encode your own 3D assets into latents following Shap-E! As Shap-E requires blender to render multi-view images for 3D encoding, you need to first install Blender version >= 3.3.1, and set the environment variable BLENDER_PATH to the path of the Blender executable. For more details, please refer to the [original repo](https://github.com/openai/shap-e). After that you can simply run to encode the 3D asset:
```buildoutcfg
python encode_3D.py --obj_path PATH_TO_YOUR_OBJ --output_dir PATH_TO_OUTPUT_DIR
```
### Step 2: Edit the latent with pre-trained Shap-Editor
After get the latent of the 3D asset, you can edit the latent within 1 secs using the pre-trained editing model or editing model you trained yourself. 
```buildoutcfg
python inference.py --obj_path PATH_TO_YOUR_OBJ --model_path PATH_TO_THE_PRETRAINED_MODEL --prompt "YOUR PROMPT"
```


## Citation

If this repo is helpful for you, please consider to cite it. Thank you! :)

```bibtex
@inproceedings{chen2024shap,
  title={SHAP-EDITOR: Instruction-guided Latent 3D Editing in Seconds},
  author={Chen, Minghao and Xie, Junyu and Laina, Iro and Vedaldi, Andrea},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26456--26466},
  year={2024}
}
```

## To Do List

- [x] Demo
- [x] Inference code
- [ ] Training code


## Acknowledgements

This research is supported by ERC-CoG UNION 101001212. Iro Laina is also partially supported by the VisualAI EPSRC grant (EP/T028572/1).

The code is largely based on [Shap-E](https://github.com/openai/shap-e). It is also inspired by worderful projects:

- [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
- [Diffusers](https://github.com/huggingface/diffusers)