from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from shap_e.models.nn.camera import (
    DifferentiableCamera,
    DifferentiableProjectiveCamera,
    get_image_coords,
    projective_camera_frame,
)
from shap_e.models.nn.meta import MetaModule
from shap_e.util.collections import AttrDict


class Renderer(MetaModule):
    """
    A rendering abstraction that can render rays and views by calling the
    appropriate models. The models are instantiated outside but registered in
    this module.
    """

    @abstractmethod
    def render_views(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        """
        Returns a backproppable rendering of a view

        :param batch: contains
            - height: Optional[int]
            - width: Optional[int]
            - inner_batch_size or ray_batch_size: Optional[int] defaults to 4096 rays

            And additionally, to specify poses with a default up direction:
            - poses: [batch_size x *shape x 2 x 3] where poses[:, ..., 0, :] are the camera
                positions, and poses[:, ..., 1, :] are the z-axis (toward the object) of
                the camera frame.
            - camera: DifferentiableCamera. Assumes the same camera position
                across batch for simplicity.  Could eventually support
                batched cameras.

            or to specify a batch of arbitrary poses:
            - cameras: DifferentiableCameraBatch of shape [batch_size x *shape].

        :param params: Meta parameters
        :param options: Optional[Dict]
        """


class RayRenderer(Renderer):
    """
    A rendering abstraction that can render rays and views by calling the
    appropriate models. The models are instantiated outside but registered in
    this module.
    """

    @abstractmethod
    def render_rays(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        """
        :param batch: has
            - rays: [batch_size x ... x 2 x 3] specify the origin and direction of each ray.
            - radii (optional): [batch_size x ... x 1] the "thickness" of each ray.
        :param options: Optional[Dict]
        """

    def render_views(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
    ) -> AttrDict:
        output = render_views_from_rays(
            self.render_rays,
            batch,
            params=params,
            options=options,
            device=self.device,
        )
        return output

    def forward(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        """
        :param batch: must contain either

            - rays: [batch_size x ... x 2 x 3] specify the origin and direction of each ray.

            or

            - poses: [batch_size x 2 x 3] where poses[:, 0] are the camera
                positions, and poses[:, 1] are the z-axis (toward the object) of
                the camera frame.
            - camera: an instance of Camera that implements camera_rays

            or

            - cameras: DifferentiableCameraBatch of shape [batch_size x *shape].

            For both of the above two options, these may be specified.
            - height: Optional[int]
            - width: Optional[int]
            - ray_batch_size or inner_batch_size: Optional[int] defaults to 4096 rays

        :param params: a dictionary of optional meta parameters.
        :param options: A Dict of other hyperparameters that could be
            related to rendering or debugging

        :return: a dictionary containing

            - channels: [batch_size, *shape, n_channels]
            - distances: [batch_size, *shape, 1]
            - transmittance: [batch_size, *shape, 1]
            - aux_losses: Dict[str, torch.Tensor]
        """

        if "rays" in batch:
            for key in ["poses", "camera", "height", "width"]:
                assert key not in batch
            return self.render_rays(batch, params=params, options=options)
        elif "poses" in batch or "cameras" in batch:
            assert "rays" not in batch
            if "poses" in batch:
                assert "camera" in batch
            else:
                assert "camera" not in batch
            return self.render_views(batch, params=params, options=options)

        raise NotImplementedError


def get_camera_from_batch(batch: AttrDict) -> Tuple[DifferentiableCamera, int, Tuple[int]]:
    if "poses" in batch:
        assert not "cameras" in batch
        batch_size, *inner_shape, n_vecs, spatial_dim = batch.poses.shape
        assert n_vecs == 2 and spatial_dim == 3
        inner_batch_size = int(np.prod(inner_shape))
        poses = batch.poses.view(batch_size * inner_batch_size, 2, 3)
        position, direction = poses[:, 0], poses[:, 1]
        camera = projective_camera_frame(position, direction, batch.camera)
    elif "cameras" in batch:
        assert not "camera" in batch
        batch_size, *inner_shape = batch.cameras.shape
        camera = batch.cameras.flat_camera
    else:
        raise ValueError(f'neither "poses" nor "cameras" found in keys: {batch.keys()}')
    if "height" in batch and "width" in batch:
        camera = camera.resize_image(batch.width, batch.height)
    return camera, batch_size, inner_shape


def append_tensor(val_list: Optional[List[torch.Tensor]], output: Optional[torch.Tensor]):
    if val_list is None:
        return [output]
    return val_list + [output]


def render_views_from_rays(
    render_rays: Callable[[AttrDict, AttrDict, AttrDict], AttrDict],
    batch: AttrDict,
    params: Optional[Dict] = None,
    options: Optional[Dict] = None,
    device: torch.device = torch.device("cuda"),
    patch_size: Optional[int] = 128,
    use_patch: bool = False,
) -> AttrDict:
    # import pdb; pdb.set_trace()
    camera, batch_size, inner_shape = get_camera_from_batch(batch)
    inner_batch_size = int(np.prod(inner_shape))

    coords = get_image_coords(camera.width, camera.height).to(device)
    coords = torch.broadcast_to(coords.unsqueeze(0), [batch_size * inner_batch_size, *coords.shape])
    rays = camera.camera_rays(coords)

    # mip-NeRF radii calculation from: https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/datasets.py#L193-L200
    directions = rays.view(batch_size, inner_batch_size, camera.height, camera.width, 2, 3)[
        ..., 1, :
    ]
    neighbor_dists = torch.linalg.norm(directions[:, :, :, 1:] - directions[:, :, :, :-1], dim=-1)
    neighbor_dists = torch.cat([neighbor_dists, neighbor_dists[:, :, :, -2:-1]], dim=3)
    radii = (neighbor_dists * 2 / np.sqrt(12)).view(batch_size, -1, 1)
    # do the patching
    if use_patch:
        print("use_patch")
        assert patch_size < camera.height
        H, W = camera.height, camera.width
        # import pdb; pdb.set_trace()
        down_scale_factor = min(H // patch_size, 4)
        rays = rays.view(batch_size*inner_batch_size, camera.height, camera.width, 2, 3)
        rays_o = rays[..., 0, :]
        rays_d = rays[..., 1, :]
        global_rays_o = torch.nn.functional.interpolate(rays_o.permute(0, 3, 1, 2),
                                                        (H // down_scale_factor, W // down_scale_factor),
                                                        mode="bilinear").permute(0, 2, 3, 1)
        global_rays_d = torch.nn.functional.interpolate(rays_d.permute(0, 3, 1, 2),
                                                        (H // down_scale_factor, W // down_scale_factor),
                                                        mode="bilinear").permute(0, 2, 3, 1)
        global_rays = torch.stack([global_rays_o, global_rays_d], dim=-2)

        global_rays = global_rays.view(batch_size, inner_batch_size * camera.height * camera.width // (down_scale_factor **2) , 2, 3)



        # rays = rays.view(batch_size, inner_batch_size * camera.height * camera.width, 2, 3)

        if isinstance(camera, DifferentiableProjectiveCamera):
            # Compute the camera z direction corresponding to every ray's pixel.
            # Used for depth computations below.
            z_directions = (
                (camera.z / torch.linalg.norm(camera.z, dim=-1, keepdim=True))
                .reshape([batch_size, inner_batch_size, 1, 3])
                .repeat(1, 1, camera.width * camera.height // down_scale_factor **2 , 1)
                .reshape(1, inner_batch_size * camera.height * camera.width // down_scale_factor ** 2, 3)
            )

        ray_batch_size = batch.get("ray_batch_size", batch.get("inner_batch_size", 1024))

        assert global_rays.shape[1] % ray_batch_size == 0
        n_batches = global_rays.shape[1] // ray_batch_size

        output_list_global = AttrDict(aux_losses=dict())
        for idx in range(n_batches):
            rays_batch = AttrDict(
                rays=global_rays[:, idx * ray_batch_size: (idx + 1) * ray_batch_size],
                radii=global_rays[:, idx * ray_batch_size: (idx + 1) * ray_batch_size],
            )
            output_global = render_rays(rays_batch, params=params, options=options)

            # output.channels.register_hook(lambda grad: print("render_rays", grad))

            if isinstance(camera, DifferentiableProjectiveCamera):
                z_batch = z_directions[:, idx * ray_batch_size: (idx + 1) * ray_batch_size]
                ray_directions = rays_batch.rays[:, :, 1]
                z_dots = (ray_directions * z_batch).sum(-1, keepdim=True)
                output_global.depth = output_global.distances * z_dots

            output_list_global = output_list_global.combine(output_global, append_tensor)

        PS = patch_size
        patch_x = torch.randint(0, W - PS, (1,)).item()
        patch_y = torch.randint(0, H - PS, (1,)).item()

        patch_rays = rays[..., patch_y: patch_y + PS, patch_x: patch_x + PS, :, :]
        patch_rays = patch_rays.reshape(batch_size, inner_batch_size * PS * PS , 2, 3)

        if isinstance(camera, DifferentiableProjectiveCamera):
            # Compute the camera z direction corresponding to every ray's pixel.
            # Used for depth computations below.
            z_directions_patch = (
                (camera.z / torch.linalg.norm(camera.z, dim=-1, keepdim=True))
                .reshape([batch_size, inner_batch_size, 1, 3])
                .repeat(1, 1, PS * PS , 1)
                .reshape(1, inner_batch_size * PS * PS, 3)
            )

        # ray_batch_size = batch.get("ray_batch_size", batch.get("inner_batch_size", 4096))
        print(ray_batch_size, patch_rays.shape[1])
        assert patch_rays.shape[1] % ray_batch_size == 0
        n_batches = patch_rays.shape[1] // ray_batch_size

        output_list = AttrDict(aux_losses=dict())
        for idx in range(n_batches):
            rays_batch = AttrDict(
                rays=patch_rays[:, idx * ray_batch_size: (idx + 1) * ray_batch_size],
                radii=patch_rays[:, idx * ray_batch_size: (idx + 1) * ray_batch_size],
            )
            output_patch = render_rays(rays_batch, params=params, options=options)

            # output.channels.register_hook(lambda grad: print("render_rays", grad))

            if isinstance(camera, DifferentiableProjectiveCamera):
                z_batch = z_directions_patch[:, idx * ray_batch_size: (idx + 1) * ray_batch_size]
                ray_directions = rays_batch.rays[:, :, 1]
                z_dots = (ray_directions * z_batch).sum(-1, keepdim=True)
                output_patch.depth = output_patch.distances * z_dots

            output_list = output_list.combine(output_patch, append_tensor)
        def _resize(val_list: List[torch.Tensor], H, W):
            val = torch.cat(val_list, dim=1)
            assert val.shape[1] == inner_batch_size * H * W
            return val.view(batch_size, *inner_shape, H, W, -1)

        def _avg(_key: str, loss_list: List[torch.Tensor]):
            return sum(loss_list) / n_batches

        output_global = AttrDict(
            {name: _resize(val_list, camera.width // down_scale_factor, camera.height // down_scale_factor) for name, val_list in output_list_global.items() if name != "aux_losses"}
        )
        output_global.aux_losses = output_list_global.aux_losses.map(_avg)

        output = AttrDict(
            {name: _resize(val_list, PS, PS) for name, val_list in output_list.items() if name != "aux_losses"}
        )
        output.aux_losses = output_list.aux_losses.map(_avg)



        valid_patch_key = []
        for key in output:
            if torch.is_tensor(output[key]):
                print(key, output[key].shape, output["channels"].shape)
                if len(output[key].shape) == len(output["channels"].shape):
                    if output[key][..., 0].shape == output["channels"][..., 0].shape:
                        valid_patch_key.append(key)
        # import pdb; pdb.set_trace()
        for key in valid_patch_key:
            if output_global[key].dtype != torch.bool:
                output_global[key] = torch.nn.functional.interpolate(
                    output_global[key].view(inner_batch_size*batch_size, camera.width // down_scale_factor, camera.height //down_scale_factor, -1).permute(0, 3, 1, 2), (H, W), mode="bilinear"
                ).permute(0, 2, 3, 1).view(batch_size, inner_batch_size, H, W, -1)
            else:
                output_global[key] = torch.nn.functional.interpolate(
                    output_global[key].view(inner_batch_size*batch_size, camera.width // down_scale_factor, camera.height //down_scale_factor, -1).permute(0, 3, 1, 2).to(torch.float32), (H, W), mode="nearest"
                ).permute(0, 2, 3, 1).view(batch_size, inner_batch_size, H, W, -1).to(torch.bool)
            output_global[key] = output_global[key].detach()
            output_global[key][
            ..., patch_y: patch_y + PS, patch_x: patch_x + PS, :
            ] = output[key]
        output = output_global


        return output

    else:
        rays = rays.view(batch_size, inner_batch_size * camera.height * camera.width, 2, 3)

        if isinstance(camera, DifferentiableProjectiveCamera):
            # Compute the camera z direction corresponding to every ray's pixel.
            # Used for depth computations below.
            z_directions = (
                (camera.z / torch.linalg.norm(camera.z, dim=-1, keepdim=True))
                .reshape([batch_size, inner_batch_size, 1, 3])
                .repeat(1, 1, camera.width * camera.height, 1)
                .reshape(1, inner_batch_size * camera.height * camera.width, 3)
            )

        ray_batch_size = batch.get("ray_batch_size", batch.get("inner_batch_size", 4096))
        assert rays.shape[1] % ray_batch_size == 0
        n_batches = rays.shape[1] // ray_batch_size

        output_list = AttrDict(aux_losses=dict())
        for idx in range(n_batches):
            rays_batch = AttrDict(
                rays=rays[:, idx * ray_batch_size : (idx + 1) * ray_batch_size],
                radii=radii[:, idx * ray_batch_size : (idx + 1) * ray_batch_size],
            )
            output = render_rays(rays_batch, params=params, options=options)

            # output.channels.register_hook(lambda grad: print("render_rays", grad))

            if isinstance(camera, DifferentiableProjectiveCamera):
                z_batch = z_directions[:, idx * ray_batch_size : (idx + 1) * ray_batch_size]
                ray_directions = rays_batch.rays[:, :, 1]
                z_dots = (ray_directions * z_batch).sum(-1, keepdim=True)
                output.depth = output.distances * z_dots

            output_list = output_list.combine(output, append_tensor)
    # for key in params:
    #     if params[key].requires_grad:
    #         params[key].register_hook(lambda grad: print("params", key, grad))
        def _resize(val_list: List[torch.Tensor]):
            val = torch.cat(val_list, dim=1)
            assert val.shape[1] == inner_batch_size * camera.height * camera.width
            return val.view(batch_size, *inner_shape, camera.height, camera.width, -1)

        def _avg(_key: str, loss_list: List[torch.Tensor]):
            return sum(loss_list) / n_batches

        output = AttrDict(
            {name: _resize(val_list) for name, val_list in output_list.items() if name != "aux_losses"}
        )
        output.aux_losses = output_list.aux_losses.map(_avg)
        return output
