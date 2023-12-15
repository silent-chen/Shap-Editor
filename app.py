import gradio as gr
import torch
import numpy as np
from functools import partial
from typing import Optional
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_mesh
import trimesh
import torch.nn as nn
import os
import random
import warnings
from huggingface_hub import hf_hub_download
import hashlib

import sys

sys.tracebacklimit = 0
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

class Blocks(gr.Blocks):

    def __init__(
            self,
            theme: str = "default",
            analytics_enabled: Optional[bool] = None,
            mode: str = "blocks",
            title: str = "Gradio",
            css: Optional[str] = None,
            **kwargs,
    ):
        self.extra_configs = {
            'thumbnail': kwargs.pop('thumbnail', ''),
            'url': kwargs.pop('url', 'https://gradio.app/'),
            'creator': kwargs.pop('creator', '@teamGradio'),
        }

        super(Blocks, self).__init__(theme, analytics_enabled, mode, title, css, **kwargs)
        warnings.filterwarnings("ignore")

    def get_config_file(self):
        config = super(Blocks, self).get_config_file()

        for k, v in self.extra_configs.items():
            config[k] = v

        return config
def optimize_all(xm, models, initial_noise, noise_start_t, diffusion, latent_model, device, prompt, instruction, rand_seed):
    state = {}
    out_gen_1, out_gen_2, out_gen_3, out_gen_4, state = generate_3d_with_shap_e(xm, diffusion, latent_model, device, prompt, rand_seed, state)
    edited_1, edited_2, edited_3, edited_4, state = _3d_editing(xm, models, diffusion, initial_noise, noise_start_t, device, instruction, rand_seed, state)
    print(state)
    return out_gen_1, out_gen_2, out_gen_3, out_gen_4, edited_1, edited_2, edited_3, edited_4
def generate_3d_with_shap_e(xm, diffusion, latent_model, device, prompt, rand_seed, state):
    set_seed(rand_seed)
    batch_size = 4
    guidance_scale = 15.0
    xm.renderer.volume.bbox_max = torch.tensor([1.0, 1.0, 1.0]).to(device)
    xm.renderer.volume.bbox_min = torch.tensor([-1.0, -1.0, -1.0]).to(device)
    xm.renderer.volume.bbox = torch.stack([xm.renderer.volume.bbox_min, xm.renderer.volume.bbox_max])

    print("prompt: ", prompt, "rand_seed: ", rand_seed, "state:",  state)
    latents = sample_latents(
        batch_size=batch_size,
        model=latent_model,
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
    prompt_hash = str(hashlib.sha256((prompt + '_' + str(rand_seed)).encode('utf-8')).hexdigest())
    mesh_path = []
    output_path = './logs'
    os.makedirs(os.path.join(output_path, 'source'), exist_ok=True)
    state['latent'] = []
    state['prompt'] = prompt
    state['rand_seed_1'] = rand_seed
    for i, latent in enumerate(latents):

        output_path_tmp = os.path.join(output_path, 'source', '{}_{}.obj'.format(prompt_hash, i))
        t_obj = decode_latent_mesh(xm, latent).tri_mesh()
        with open(output_path_tmp, 'w') as f:
            t_obj.write_obj(f)

        mesh = trimesh.load_mesh(output_path_tmp)
        angle = np.radians(180)
        axis = [0, 1, 0]
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
        mesh.apply_transform(rotation_matrix)
        angle = np.radians(90)
        axis = [1, 0, 0]
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
        mesh.apply_transform(rotation_matrix)
        output_path_tmp = os.path.join(output_path, 'source', '{}_{}.obj'.format(prompt_hash, i))
        mesh.export(output_path_tmp)
        state['latent'].append(latent.clone().detach())
        mesh_path.append(output_path_tmp)

    return mesh_path[0], mesh_path[1], mesh_path[2], mesh_path[3], state

def _3d_editing(xm, models, diffusion, initial_noise, start_t, device, instruction, rand_seed, state):
    set_seed(rand_seed)
    mesh_path = []
    prompt = state['prompt']
    rand_seed_1 = state['rand_seed_1']
    print("prompt: ", prompt, "rand_seed: ", rand_seed, "instruction:", instruction, "state:",  state)
    prompt_hash = str(hashlib.sha256((prompt + '_' + str(rand_seed_1) + '_' + instruction + '_' + str(rand_seed)).encode('utf-8')).hexdigest())
    if 'santa' in instruction:
        e_type = 'santa_hat'
    elif 'rainbow' in instruction:
        e_type = 'rainbow'
    elif 'gold' in instruction:
        e_type = 'golden'
    elif 'lego' in instruction:
        e_type = 'lego'
    elif 'wooden' in instruction:
        e_type = 'wooden'
    elif 'cyber' in instruction:
        e_type = 'cyber'

    # import pdb; pdb.set_trace()
    model = models[e_type].to(device)
    noise_initial = initial_noise[e_type].to(device)
    noise_start_t = start_t[e_type]
    general_save_path = './logs/edited'
    os.makedirs(general_save_path, exist_ok=True)
    for i, latent in enumerate(state['latent']):
        latent = latent.to(device)
        text_embeddings_clip = model.cached_model_kwargs(1, dict(texts=[instruction]))
        print("shape of latent: ", latent.clone().unsqueeze(0).shape, "instruction: ", instruction)
        ref_latent = latent.clone().unsqueeze(0)
        t_1 = torch.randint(noise_start_t, noise_start_t + 1, (1,), device=device).long()

        noise_input = diffusion.q_sample(ref_latent, t_1, noise=noise_initial)
        out_1 = diffusion.p_mean_variance(model, noise_input, t_1, clip_denoised=True,
                                          model_kwargs=text_embeddings_clip,
                                          condition_latents=ref_latent)

        updated_latents = out_1['pred_xstart']

        if 'santa' in instruction:
            xm.renderer.volume.bbox_max = torch.tensor([1.0, 1.0, 1.25]).to(device)
            xm.renderer.volume.bbox_min = torch.tensor([-1.0, -1.0, -1]).to(device)
            xm.renderer.volume.bbox = torch.stack([xm.renderer.volume.bbox_min, xm.renderer.volume.bbox_max])

        else:
            xm.renderer.volume.bbox_max = torch.tensor([1.0, 1.0, 1.0]).to(device)
            xm.renderer.volume.bbox_min = torch.tensor([-1.0, -1.0, -1.0]).to(device)
            xm.renderer.volume.bbox = torch.stack([xm.renderer.volume.bbox_min, xm.renderer.volume.bbox_max])

        for latent_idx, updated_latent in enumerate(updated_latents):
            output_path = os.path.join(general_save_path, '{}_{}.obj'.format(prompt_hash, i))

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

            output_path = os.path.join(general_save_path, '{}_{}.obj'.format(prompt_hash, i))
            mesh.export(output_path)
            mesh_path.append(output_path)
    return mesh_path[0], mesh_path[1], mesh_path[2], mesh_path[3], state
def main():

    css = """
    #img2img_image, #img2img_image > .fixed-height, #img2img_image > .fixed-height > div, #img2img_image > .fixed-height > div > img
    {
        height: var(--height) !important;
        max-height: var(--height) !important;
        min-height: var(--height) !important;
    }
    #paper-info a {
        color:#008AD7;
        text-decoration: none;
    }
    #paper-info a:hover {
        cursor: pointer;
        text-decoration: none;
    }

    .tooltip {
        color: #555;
        position: relative;
        display: inline-block;
        cursor: pointer;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 400px;
        background-color: #555;
        color: #fff;
        text-align: center;
        padding: 5px;
        border-radius: 5px;
        position: absolute;
        z-index: 1; /* Set z-index to 1 */
        left: 10px;
        top: 100%;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
        z-index: 9999; /* Set a high z-index value when hovering */
    }


    """

    rescale_js = """
    function(x) {
        const root = document.querySelector('gradio-app').shadowRoot || document.querySelector('gradio-app');
        let image_scale = parseFloat(root.querySelector('#image_scale input').value) || 1.0;
        const image_width = root.querySelector('#img2img_image').clientWidth;
        const target_height = parseInt(image_width * image_scale);
        document.body.style.setProperty('--height', `${target_height}px`);
        root.querySelectorAll('button.justify-center.rounded')[0].style.display='none';
        root.querySelectorAll('button.justify-center.rounded')[1].style.display='none';
        return x;
    }
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_model = load_model('text300M', device=device)
    xm = load_model('transmitter', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    freeze_params(xm.parameters())
    models = dict()
    initial_noise = dict()
    noise_start_t = dict()
    editing_types = ['rainbow', 'santa_hat', 'lego', 'golden', 'wooden', 'cyber']

    for editing_type in editing_types:
        tmp_model = load_model('text300M', device=device)
        with torch.no_grad():
            new_proj = nn.Linear(1024 * 2, 1024, device=device, dtype=tmp_model.wrapped.input_proj.weight.dtype)
            new_proj.weight = nn.Parameter(torch.zeros_like(new_proj.weight))
            new_proj.weight[:, :1024].copy_(tmp_model.wrapped.input_proj.weight)  #
            new_proj.bias = nn.Parameter(torch.zeros_like(new_proj.bias))
            new_proj.bias[:1024].copy_(tmp_model.wrapped.input_proj.bias)
            tmp_model.wrapped.input_proj = new_proj

        ckp = torch.load(hf_hub_download(repo_id='silentchen/Shap_Editor', subfolder='single', filename='{}.pt'.format(editing_type)), map_location='cpu')
        tmp_model.load_state_dict(ckp['model'])
        noise_initial = ckp['initial_noise']['noise'].to(device)
        initial_noise[editing_type] = noise_initial
        noise_start_t[editing_type] = ckp['t_start']
        models[editing_type] = tmp_model

    with Blocks(
            css=css,
            analytics_enabled=False,
            title="SHAPE-EDITOR demo",
    ) as demo:
        description = """<p style="text-align: center; font-weight: bold;">
            <span style="font-size: 28px"> <span style="font-size: 140%">S</span>HAP-<span style="font-size: 140%">E</span>DITOR: Instruction-guided <br> Latent 3D Editing in Seconds</span>
            <br>
            <span style="font-size: 18px" id="paper-info">
                [<a href="https://silent-chen.github.io/Shap-Editor/" target="_blank">Project Page</a>]
                [<a href="http://arxiv.org/abs/2312.09246" target="_blank">Paper</a>]
                [<a href="https://github.com/silent-chen/Shap-Editor" target="_blank">GitHub</a>]
            </span>
        </p>
        """
        state = gr.State({})
        gr.HTML(description)
        with gr.Column():
            with gr.Column():
                gr.HTML('<span style="font-size: 20px; font-weight: bold">Step 1: generate original 3D objects using Shap-E.</span>')
                prompt = gr.Textbox(
                    label="Text prompt for initial 3D generation", lines=1
                )
                gen_btn = gr.Button(value='Generate', scale=1)


            with gr.Column():
                gr.HTML('<span style="font-size: 20px; font-weight: bold">Generated 3D objects</span>')
                with gr.Row():
                    out_gen_1 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], visible=True, label="3D Model 1 (step 1)")
                    out_gen_2 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], visible=True, label="3D Model 2 (step 1)")
                    out_gen_3 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], visible=True, label="3D Model 3 (step 1)")
                    out_gen_4 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  visible=True, label="3D Model 4 (step 1)")

            with gr.Column(scale=1):
                gr.HTML('<span style="font-size: 20px; font-weight: bold">Step 2: apply 3D editing with S</span>HAP-<span style="font-size: 140%">E</span>DITOR.</span>')

                editing_choice = gr.Dropdown(
                    ["Add a santa hat to it", "Make it look like made of gold", "Make the color of it look like rainbow", "Make it in cyberpunk style", "Make it wooden", "Make it look like make of lego"], value='Add a santa hat to it', multiselect=False, label="Editing effects", info="Select specific editing you want to apply!"
                ),
                apply_btn = gr.Button(value='Editing', scale=1)

            with gr.Column(scale=3):
                gr.HTML('<span style="font-size: 20px; font-weight: bold">Edited 3D objects</span>')
                with gr.Row():
                    edited_1 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], visible=True, label="3D Model 1 (step 2)")
                    edited_2 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], visible=True, label="3D Model 2 (step 2)")
                    edited_3 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], visible=True, label="3D Model 3 (step 2)")
                    edited_4 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], visible=True, label="3D Model 4 (step 2)")


            with gr.Accordion("Advanced Options", open=False):
                rand_seed = gr.Slider(minimum=0, maximum=1000, step=1, value=445, label="Random seed")

            gen_btn.click(
                fn=partial(generate_3d_with_shap_e, xm, diffusion, latent_model, device),
                inputs=[prompt, rand_seed, state],
                outputs=[out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
                queue=False)

            apply_btn.click(
                fn=partial(_3d_editing, xm, models, diffusion, initial_noise, noise_start_t, device),
                inputs=[
                    editing_choice[0], rand_seed, state
                ],
                outputs=[edited_1, edited_2, edited_3, edited_4, state],
                queue=True
            )
        print("Generate examples...")
        with gr.Column():
            gr.Examples(
                examples=[
                    [   "a corgi",
                        "Make the color of it look like rainbow",
                        456,
                    ],
                    ["a penguin",
                     "Make it look like make of lego",
                     214,
                     ],
                ],
                inputs=[prompt, editing_choice[0], rand_seed],
                outputs=[out_gen_1, out_gen_2, out_gen_3, out_gen_4, edited_1, edited_2, edited_3, edited_4],
                fn=partial(optimize_all, xm, models, initial_noise, noise_start_t, diffusion, latent_model, device),
                cache_examples=True,
            )


    demo.queue(max_size=10, api_open=False)
    demo.launch(share=True, show_api=False, show_error=True)

if __name__ == '__main__':
    main()