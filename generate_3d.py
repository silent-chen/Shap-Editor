import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
import argparse
import os
import trimesh
import numpy as np

def generate_latent(xm, model, diffusion, prompt, guidance_scale, batch_size, output_dir, export_mesh=False):

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    output_dir = os.path.join(output_dir, '_'.join(prompt.split(' ')))
    os.makedirs(output_dir, exist_ok=True)
    for idx, latent in enumerate(latents):
        torch.save(latent, f'{output_dir}/{idx}.pth')
        if export_mesh:
            t = decode_latent_mesh(xm, latent).tri_mesh()
            mesh_output_path = f'{output_dir}/{idx}.obj'
            with open(mesh_output_path, 'w') as f:
                t.write_obj(f)
            mesh = trimesh.load_mesh(mesh_output_path)

            angle = np.radians(180)
            axis = [0, 1, 0]

            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
            mesh.apply_transform(rotation_matrix)
            angle = np.radians(90)
            axis = [1, 0, 0]

            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
            mesh.apply_transform(rotation_matrix)

            mesh.export(mesh_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='a shark')
    parser.add_argument("--output_dir", type=str, default='./output/example/', help="path to output directory")
    parser.add_argument("--guidance_scale", type=float, default=15.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--export_mesh', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    os.makedirs(args.output_dir, exist_ok=True)
    generate_latent(xm, model, diffusion, prompt=args.prompt, guidance_scale=args.guidance_scale, batch_size=args.batch_size, output_dir=args.output_dir, export_mesh=args.export_mesh)


