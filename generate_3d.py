import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
import argparse
import os
def generate_latent(xm, model, diffusion, prompt, guidance_scale, batch_size, output_dir):

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='a shark')
    parser.add_argument("--output_dir", type=str, default='./output/example/', help="path to output directory")
    parser.add_argument("--guidance_scale", type=float, default=15.0)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    os.makedirs(args.output_dir, exist_ok=True)
    generate_latent(xm, model, diffusion, prompt=args.prompt, guidance_scale=args.guidance_scale, batch_size=args.batch_size, output_dir=args.output_dir)


