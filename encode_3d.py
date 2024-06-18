import torch

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
import argparse
import os

def encode_3d(model, obj_path, output_dir):


    # This may take a few minutes, since it requires rendering the model twice
    # in two different modes.
    obj_name = os.path.basename(obj_path).split('.')[0]
    os.makedirs(f"example_cache/{obj_name}/cached", exist_ok=True)
    batch = load_or_create_multimodal_batch(
        device,
        model_path=obj_path,
        mv_light_mode="basic",
        mv_image_size=256,
        cache_dir=f"example_cache/{obj_name}/cached",
        verbose=True,  # this will show Blender output during renders
    )
    with torch.no_grad():
        latent = model.encoder.encode_to_bottleneck(batch)
    torch.save(latent, os.path.join(output_dir, f'{obj_name}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True,  help="path to the 3D obj")
    parser.add_argument("--output_dir", type=str, default='./output/example', help="path to output directory")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    os.makedirs(args.output_dir, exist_ok=True)
    encode_3d(xm, obj_path=args.obj_path, output_dir=args.output_dir)


