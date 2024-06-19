import gradio as gr
import torch
import numpy as np
from functools import partial
from typing import Optional
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_mesh, decode_latent_images
import trimesh
import torch.nn as nn
import os
import random
import argparse

import sys

def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def _3d_editing(prompt, xm, model, diffusion, initial_noise, noise_start_t, device, latent_path=None, rand_seed=42, output_dir=None, render_size=512, render_mode='stf', export_mesh=True, export_render=False):
    set_seed(rand_seed)
    general_save_path = output_dir

    os.makedirs(general_save_path, exist_ok=True)

    latent = torch.load(latent_path, map_location='cpu').to(device).squeeze()
    text_embeddings_clip = model.cached_model_kwargs(1, dict(texts=[prompt]))
    ref_latent = latent.clone().unsqueeze(0)
    t_1 = torch.randint(noise_start_t, noise_start_t + 1, (1,), device=device).long()

    noise_input = diffusion.q_sample(ref_latent, t_1, noise=initial_noise)
    out_1 = diffusion.p_mean_variance(model, noise_input, t_1, clip_denoised=True,
                                      model_kwargs=text_embeddings_clip,
                                      condition_latents=ref_latent)

    updated_latents = out_1['pred_xstart']

    for latent_idx, updated_latent in enumerate(updated_latents):
        output_path = os.path.join(general_save_path, '{}_{}.obj'.format('_'.join(prompt.split(' ')), latent_idx))
        if export_render:
            cameras = create_pan_cameras(render_size, device)
            images = decode_latent_images(xm, updated_latent, cameras, rendering_mode=render_mode)
            images[0].save(
                os.path.join(general_save_path, '{}_{}.gif'.format('_'.join(prompt.split(' ')), latent_idx)), format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0
            )

        if export_mesh:
            t = decode_latent_mesh(xm, updated_latent).tri_mesh()
            with open(output_path, 'w') as f:
                t.write_obj(f)
            mesh = trimesh.load_mesh(output_path)

            angle = np.radians(180)
            axis = [0, 1, 0]

            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
            mesh.apply_transform(rotation_matrix)
            angle = np.radians(90)
            axis = [1, 0, 0]

            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
            mesh.apply_transform(rotation_matrix)

            mesh.export(output_path)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_path", type=str, help="path to the 3D latent")
    parser.add_argument("--output_dir", type=str, default='./output/example', help="path to output directory")
    parser.add_argument("--model_path", type=str, help="path to pretrained model")
    parser.add_argument("--prompt", type=str, default='turn it into gold')
    parser.add_argument("--rand_seed", type=int, default=42)
    parser.add_argument('--export_mesh', action='store_true')
    parser.add_argument("--export_render",  action='store_true')
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument("--render_mode", type=str, default='stf')


    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the component for Shap-Editor
    xm = load_model('transmitter', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    freeze_params(xm.parameters())
    model = load_model('text300M', device=device)

    # Consider the initial noise and additonal channels
    with torch.no_grad():
        new_proj = nn.Linear(1024 * 2, 1024, device=device, dtype=model.wrapped.input_proj.weight.dtype)
        new_proj.weight = nn.Parameter(torch.zeros_like(new_proj.weight))
        new_proj.weight[:, :1024].copy_(model.wrapped.input_proj.weight)  #
        new_proj.bias = nn.Parameter(torch.zeros_like(new_proj.bias))
        new_proj.bias[:1024].copy_(model.wrapped.input_proj.bias)
        model.wrapped.input_proj = new_proj

    ckp = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckp['model'])
    initial_noise = ckp['initial_noise']['noise'].to(device)
    noise_start_t = ckp['t_start']

    # load the source latent
    _3d_editing(prompt=args.prompt,
                xm = xm,
                model = model,
                diffusion = diffusion,
                initial_noise = initial_noise,
                noise_start_t = noise_start_t,
                device = device,
                rand_seed = args.rand_seed,
                latent_path = args.latent_path,
                output_dir= args.output_dir,
                render_mode = args.render_mode,
                render_size = args.render_size,
                export_mesh = args.export_mesh,
                export_render = args.export_render
    )

if __name__ == '__main__':
    main()