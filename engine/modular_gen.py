import warnings
warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', 'To use flash-attn v3, please use the following commands to install', category=UserWarning)
warnings.filterwarnings('ignore', 'incompatible copy of pydevd already imported', category=UserWarning)

import argparse
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
from pathlib import Path


from utils.loading import get_model
from utils.loading import get_necessary_data, load_config
from utils.logger import print_with_prefix
from models.lightningdit import LightningDiT
from models.dinov2 import get_dino_v2_model_256
from utils.data_utils import load_images_from_local, create_image_grid
from utils.pipeline import FeatureExtractionPipeline, SamplingPipeline, DecodingPipeline, InterpolationPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------------------------------------------------------------------------------- #
#  1) new_snippet_generation / interpolate_generation 
# -------------------------------------------------------------------------------------- #

def new_snippet_generation(
    model,
    device,
    sample_fn,
    vae,
    latent_mean,
    latent_std,
    latent_multiplier,
    latent_size,
    image_path: str,
    n_generated_images: int = 14,
    encoder_model=None
):
    """
    Generate multiple images for each input using CFG sampling with extracted features.
    This function has been refactored to use the pipeline approach internally, 
    but we keep the function signature and final image saving process the same.

    Args:
        model: Diffusion model.
        device: Torch device.
        sample_fn: Diffusion sampling function.
        vae: VAE container (has a .model attribute).
        latent_mean: Latent mean for post-processing.
        latent_std: Latent std for post-processing.
        latent_multiplier: The multiplier used in latents.
        image_path (str): Path to a directory with images.
        n_generated_images (int): Number of images to generate per input image.
        encoder_model: Pre-trained feature extractor (e.g., DINO).
    """
    # Pipelines
    feature_extraction = FeatureExtractionPipeline(encoder_model, device)
    sampling_pipeline = SamplingPipeline(model, sample_fn, latent_mean, latent_std, latent_multiplier)
    decoding_pipeline = DecodingPipeline(vae)

    image_dir = Path(image_path)
    image_files = sorted([p for p in image_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])

    # Load images
    ori_imgs = load_images_from_local(image_files).to(device)
    # Extract features
    cs = feature_extraction(ori_imgs)

    # List for final composition
    all_images = []

    for single_img_tensor, single_feature, img_path in zip(ori_imgs, cs, image_files):
        # Add original image (resized to 256x256) to the collage
        original_img = Image.open(img_path).convert("RGB").resize((256, 256))
        all_images.append(original_img)

        # Expand single_feature to [n_generated_images, feature_dim]
        expanded_features = single_feature.unsqueeze(0).repeat(n_generated_images, 1) # [n_generated_images, feature_dim]

        # Run sampling
        latents = sampling_pipeline(
            condition_vectors=expanded_features,
            cfg_scale=7.75,
            cfg_interval=True,
            cfg_interval_start=0.89,
            n_images=n_generated_images,
            latent_size=latent_size,  # or latent_size from config
            device=device
        )
    
        # Decode latents
        generated_imgs = decoding_pipeline(latents)
        all_images.extend(generated_imgs)

    # Create final grid: one original + n_generated_images in a row
    img_width, img_height = all_images[0].size
    columns = n_generated_images + 1
    grid = create_image_grid(all_images, img_width, img_height, columns)

    # Save image
    output_dir = Path("assets/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.save(output_dir / "new_snippet_generation.png")


def interpolate_generation(
    model,
    device,
    sample_fn,
    vae,
    latent_mean,
    latent_std,
    latent_multiplier,
    latent_size,
    image_paths,
    n_generated_images: int = 6,
    batch_size: int = 8,
    encoder_model=None,
    interpolation_method: str = "spherical"
):
    """
    Interpolate between two sets of images (1-to-1 matched), then sample and decode the results.
    This function also uses the pipeline approach internally.

    Args:
        model: Diffusion model.
        device: Torch device.
        sample_fn: Diffusion sampling function.
        vae: VAE container (has a .model attribute).
        latent_mean: Latent mean for post-processing.
        latent_std: Latent std for post-processing.
        latent_multiplier: The multiplier used in latents.
        image_paths: A list or string representing two directories with matched images.
        n_generated_images (int): Number of interpolation steps (excluding or including endpoints).
        batch_size (int): The batch size for sampling.
        encoder_model: Pre-trained feature extractor.
        interpolation_method (str): "linear" or "spherical".
    """
    # Pipelines
    feature_extraction = FeatureExtractionPipeline(encoder_model, device)
    interpolation_pipeline = InterpolationPipeline(method=interpolation_method, total_steps=n_generated_images)
    sampling_pipeline = SamplingPipeline(model, sample_fn, latent_mean, latent_std, latent_multiplier)
    decoding_pipeline = DecodingPipeline(vae)

    # Parse image_paths
    if isinstance(image_paths, str):
        # e.g. "['assets/A','assets/B']" or "assets/A,assets/B"
        paths = image_paths.strip("[]").split(",")
        paths = [p.strip() for p in paths]
    else:
        paths = image_paths

    assert len(paths) == 2, "You must provide exactly two directories for interpolation."
    path_a, path_b = Path(paths[0]), Path(paths[1])

    # Load matched images
    image_files_a = sorted([p for p in path_a.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])
    image_files_b = sorted([p for p in path_b.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])
    assert len(image_files_a) == len(image_files_b), "Image pairs must be matched 1:1."

    imgs_a = load_images_from_local(image_files_a).to(device)
    imgs_b = load_images_from_local(image_files_b).to(device)

    # Extract features
    feats_a = feature_extraction(imgs_a)
    feats_b = feature_extraction(imgs_b)

    all_rows = []
    img_w, img_h = 256, 256  # for final grid
    num_pairs = len(image_files_a)

    for i in range(num_pairs):
        # Original images for left/right ends
        left_img = imgs_a[i] * 255
        right_img = imgs_b[i] * 255

        # Interpolate features
        feats_list = interpolation_pipeline(feats_a[i], feats_b[i])  # list of features
        all_interpolated_imgs = []

        # Process interpolation in batches
        for start_idx in range(0, len(feats_list), batch_size):
            end_idx = min(start_idx + batch_size, len(feats_list))
            batch_feats = torch.stack(feats_list[start_idx:end_idx], dim=0)  # shape [B, feat_dim]

            # Sample
            latents = sampling_pipeline(
                condition_vectors=batch_feats,
                cfg_scale=7.75,
                cfg_interval=True,
                cfg_interval_start=0.89,
                n_images=1,
                latent_size=latent_size,
                device=device
            )
            # Decode
            decoded = decoding_pipeline(latents)
            all_interpolated_imgs.extend(decoded)

        # Convert left and right image to PIL for final row
        left_pil = Image.fromarray(left_img.permute(1,2,0).cpu().numpy().astype(np.uint8))
        left_pil = left_pil.resize((img_w, img_h))
        right_pil = Image.fromarray(right_img.permute(1,2,0).cpu().numpy().astype(np.uint8))
        right_pil = right_pil.resize((img_w, img_h))

        # Build row: left -> interpolation results -> right
        row_imgs = [left_pil] + all_interpolated_imgs + [right_pil]
        for ri in row_imgs:
            all_rows.append(ri)

    n_cols = len(feats_list) + 2  # each row has #interpolation steps + 2 endpoints
    final_grid = create_image_grid(all_rows, img_w, img_h, n_cols)

    # Save
    output_dir = Path("assets/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_grid.save(output_dir / f"interpolation_{interpolation_method}.png")

    return final_grid


# -------------------------------------------------------------------------------------- #
#  main driver function: do_sample
# -------------------------------------------------------------------------------------- #

@torch.no_grad()
def do_sample(train_config, model: LightningDiT, image_path: str = None, args=None):
    """
    Run sampling. This function is kept mostly as-is, with minor adjustments 
    to call the refactored pipeline-based new_snippet_cfg or interpolate_semantic_cfg.
    """
    assert torch.cuda.is_available(), "Sampling with at least one GPU is required."
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Retrieve necessary data
    timestep_shift, latent_size, transport, sampler, sample_fn, vae, latent_mean, latent_std, latent_multiplier, _ = get_necessary_data(train_config)

    # Move relevant modules/tensors to device
    model, latent_mean, latent_std, vae.model = map(lambda x: x.to(device), (model, latent_mean, latent_std, vae.model))

    # Here we call either snippet or interpolation. 
    # For demonstration, assume `image_path` can be one or two directories. 
    # The "mode" can be recognized by how many directories passed or by config.

    # Example logic:
    if args.mode == 'external':
        # Single folder => new_snippet_cfg
        new_snippet_generation(
            model=model,
            device=device,
            sample_fn=sample_fn,
            vae=vae,
            latent_mean=latent_mean,
            latent_std=latent_std,
            latent_multiplier=latent_multiplier,
            latent_size=latent_size,
            image_path=image_path,
            n_generated_images=14,
            encoder_model=get_dino_v2_model_256()
        )
    elif args.mode == 'internal':
        # If we detect multiple directories => interpolation
        interpolate_generation(
            model=model,
            device=device,
            sample_fn=sample_fn,
            vae=vae,
            latent_mean=latent_mean,
            latent_std=latent_std,
            latent_multiplier=latent_multiplier,
            latent_size=latent_size,
            image_paths=image_path,  # e.g. "assets/A, assets/B"
            n_generated_images=6,
            batch_size=8,
            encoder_model=get_dino_v2_model_256(),
            interpolation_method=args.method
        )
    else:
        raise ValueError("Invalid mode. Must be 'external' or 'internal'.")

    # Clean up
    del latent_mean, latent_std, latent_multiplier, device, sample_fn, transport, sampler, timestep_shift, latent_size
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/samplers/sde_xl_cfg.yaml')
    parser.add_argument('--image_path', type=str, default='assets/novel_images')
    parser.add_argument('--mode', type=str, default='external')
    parser.add_argument('--method', type=str, default='mask')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_config = load_config(args.config)
    torch.manual_seed(args.seed)

    # Load model from checkpoint
    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    model = get_model(train_config)
    model.eval()

    # Start sampling
    do_sample(train_config, model=model, image_path=args.image_path, args=args)
    print_with_prefix('Sampling Done!')
