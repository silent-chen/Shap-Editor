from typing import Any, Callable, Dict, Optional, List

import torch
import torch.nn as nn

from .gaussian_diffusion import GaussianDiffusion
from .k_diffusion import karras_sample, karras_sample_addition_condition

DEFAULT_KARRAS_STEPS = 64
DEFAULT_KARRAS_SIGMA_MIN = 1e-3
DEFAULT_KARRAS_SIGMA_MAX = 160
DEFAULT_KARRAS_S_CHURN = 0.0


def uncond_guide_model(
    model: Callable[..., torch.Tensor], scale: float
) -> Callable[..., torch.Tensor]:

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        cond_out, uncond_out = torch.chunk(model_out, 2, dim=0)
        cond_out = uncond_out + scale * (cond_out - uncond_out)
        return torch.cat([cond_out, cond_out], dim=0)

    return model_fn


def sample_latents(
    *,
    batch_size: int,
    model: nn.Module,
    diffusion: GaussianDiffusion,
    model_kwargs: Dict[str, Any],
    guidance_scale: float,
    clip_denoised: bool,
    use_fp16: bool,
    use_karras: bool,
    karras_steps: int,
    sigma_min: float,
    sigma_max: float,
    s_churn: float,
    device: Optional[torch.device] = None,
    progress: bool = False,
    initial_noise: Optional[torch.Tensor] = None,
) -> (torch.Tensor, List[torch.Tensor]):
    sample_shape = (batch_size, model.d_latent)

    if device is None:
        device = next(model.parameters()).device

    if hasattr(model, "cached_model_kwargs"):
        model_kwargs = model.cached_model_kwargs(batch_size, model_kwargs)
    if guidance_scale != 1.0 and guidance_scale != 0.0:
        for k, v in model_kwargs.copy().items():
            # print(k, v.shape)
            model_kwargs[k] = torch.cat([v, torch.zeros_like(v)], dim=0)

    sample_shape = (batch_size, model.d_latent)
    with torch.autocast(device_type=device.type, enabled=use_fp16):
        if use_karras:
            samples, sample_sequence = karras_sample(
                diffusion=diffusion,
                model=model,
                shape=sample_shape,
                steps=karras_steps,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                s_churn=s_churn,
                guidance_scale=guidance_scale,
                progress=progress,
                initial_noise=initial_noise,
            )
        else:
            internal_batch_size = batch_size
            if guidance_scale != 1.0:
                model = uncond_guide_model(model, guidance_scale)
                internal_batch_size *= 2
            samples = diffusion.p_sample_loop(
                model,
                shape=(internal_batch_size, *sample_shape[1:]),
                model_kwargs=model_kwargs,
                device=device,
                clip_denoised=clip_denoised,
                progress=progress,
            )

    return samples


def sample_latents_with_additional_latent(
    *,
    batch_size: int,
    model: nn.Module,
    diffusion: GaussianDiffusion,
    model_kwargs: Dict[str, Any],
    text_guidance_scale: float,
    image_guidance_scale: float,
    clip_denoised: bool,
    use_fp16: bool,
    use_karras: bool,
    karras_steps: int,
    sigma_min: float,
    sigma_max: float,
    s_churn: float,
    device: Optional[torch.device] = None,
    progress: bool = False,
    condition_latent: Optional[torch.Tensor] = None,
) -> (torch.Tensor, List[torch.Tensor]):

    if device is None:
        device = next(model.parameters()).device

    if hasattr(model, "cached_model_kwargs"):
        model_kwargs = model.cached_model_kwargs(batch_size, model_kwargs)
    if (text_guidance_scale != 1.0 and text_guidance_scale != 0.0) or (image_guidance_scale != 1.0 and image_guidance_scale != 0.0):
        for k, v in model_kwargs.copy().items():
            # print(k, v.shape)
            model_kwargs[k] = torch.cat([v, torch.zeros_like(v), torch.zeros_like(v)], dim=0)
            condition_latent = torch.cat([condition_latent, condition_latent, torch.zeros_like(condition_latent)], dim=0)

    sample_shape = (batch_size, model.d_latent)
    # print("sample_shape", sample_shape)
    with torch.autocast(device_type=device.type, enabled=use_fp16):
        if use_karras:
            samples, samples_squence = karras_sample_addition_condition(
                diffusion=diffusion,
                model=model,
                shape=sample_shape,
                steps=karras_steps,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                s_churn=s_churn,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                progress=progress,
                condition_latent=condition_latent,
            )
        else:
            internal_batch_size = batch_size
            if text_guidance_scale != 1.0:
                model = uncond_guide_model(model, text_guidance_scale)
                internal_batch_size *= 2
            samples = diffusion.p_sample_loop(
                model,
                shape=(internal_batch_size, *sample_shape[1:]),
                model_kwargs=model_kwargs,
                device=device,
                clip_denoised=clip_denoised,
                progress=progress,
            )

    return samples