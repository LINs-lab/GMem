import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.lightningdit import LightningDiT
from models.dinov2 import get_dino_v2_representation

# -------------------------------------------------------------------------------------- #
#   Pipeline Classes: feature extraction / interpolation / sampling / decoding / grid
# -------------------------------------------------------------------------------------- #

class FeatureExtractionPipeline:
    """
    A pipeline class for extracting features from images using a pre-trained encoder (e.g., DINO).
    """
    def __init__(self, encoder_model, device):
        """
        Args:
            encoder_model: A pre-trained image encoder model (e.g., DINO).
            device: Torch device (e.g., 'cpu' or 'cuda').
        """
        self.encoder = encoder_model.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, images: torch.Tensor):
        """
        Extract features from a batch of images.

        Args:
            images (torch.Tensor): A batch of preprocessed images, shape [B, 3, H, W].

        Returns:
            torch.Tensor: The extracted CLS-token features, shape [B, feature_dim].
        """
        # raw_images -> get_dino_v2_representation(...) returns (representation, cls_token)
        _, cls_token = get_dino_v2_representation(raw_images=images, model=self.encoder)
        return F.normalize(cls_token, p=2, dim=-1)  


def linear_interpolation(feats_a: torch.Tensor, feats_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Linear interpolation between feats_a and feats_b.

    Args:
        feats_a (torch.Tensor): Feature A, shape [feature_dim].
        feats_b (torch.Tensor): Feature B, shape [feature_dim].
        alpha (float): Interpolation coefficient between 0 and 1.

    Returns:
        torch.Tensor: Interpolated feature, shape [feature_dim].
    """
    return (1 - alpha) * feats_a + alpha * feats_b


def spherical_interpolation(feats_a: torch.Tensor, feats_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical interpolation between feats_a and feats_b.

    Args:
        feats_a (torch.Tensor): Feature A, shape [feature_dim].
        feats_b (torch.Tensor): Feature B, shape [feature_dim].
        alpha (float): Interpolation coefficient between 0 and 1.

    Returns:
        torch.Tensor: Interpolated feature, shape [feature_dim].
    """

    dot = torch.sum(feats_a * feats_b, dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Prevent potential zero-div error
    mask = sin_theta > 1e-6
    sin_theta[~mask] = 1e-6

    part_a = torch.sin((1 - alpha) * theta) / sin_theta
    part_b = torch.sin(alpha * theta) / sin_theta

    return part_a * feats_a + part_b * feats_b

def concat_interpolation(feats_a: torch.Tensor, feats_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Concat interpolation between feats_a and feats_b.

    Args:
        feats_a (torch.Tensor): Feature A, shape [feature_dim].
        feats_b (torch.Tensor): Feature B, shape [feature_dim].
        alpha (float): Interpolation coefficient between 0 and 1.

    Returns:
        torch.Tensor: Interpolated feature, shape [feature_dim].
    """
    first_half = feats_a[:feats_a.shape[0]//2]
    second_half = feats_b[feats_b.shape[0]//2:]
    return torch.cat([first_half, second_half], dim=0)

def mask_interpolation(feats_a: torch.Tensor, feats_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Substract interpolation between feats_a and feats_b.

    Args:
        feats_a (torch.Tensor): Feature A, shape [feature_dim].
        feats_b (torch.Tensor): Feature B, shape [feature_dim].
        alpha (float): Useless, just for compatibility.

    Returns:
        torch.Tensor: Interpolated feature, shape [feature_dim].
    """

    mask = torch.rand(feats_a.shape, device=feats_a.device) > 0.5
    return torch.where(mask, feats_a, feats_b)

class InterpolationPipeline:
    """
    A pipeline class for interpolating between two feature vectors (or batches).
    """
    def __init__(self, method: str = "concat", total_steps: int = 6):
        """
        Args:
            method (str): Interpolation method, either "linear", "spherical", "concat" or "mask".
            total_steps (int): Number of interpolation steps (excluding the endpoints or including, up to design).
        """
        self.method = method
        self.total_steps = total_steps

    def __call__(self, feats_a: torch.Tensor, feats_b: torch.Tensor) -> list:
        """
        Interpolate between feats_a and feats_b over self.total_steps points.

        Args:
            feats_a (torch.Tensor): Feature A, shape [feature_dim] or [N, feature_dim].
            feats_b (torch.Tensor): Feature B, shape [feature_dim] or [N, feature_dim].

        Returns:
            list: A list of interpolated features, each is a torch.Tensor of shape [feature_dim] or [N, feature_dim].
        """
        # If feats_a is 2D, we assume feats_a and feats_b have shape [N, feature_dim], do elementwise.
        # We will generate 'total_steps' equally spaced alphas.
        alphas = torch.linspace(0, 1, self.total_steps, device=feats_a.device)
        result = []
        for alpha in alphas:
            if self.method == "linear":
                interp_feats = linear_interpolation(feats_a, feats_b, alpha)
            elif self.method == "spherical":
                interp_feats = spherical_interpolation(feats_a, feats_b, alpha)
            elif self.method == "concat":
                interp_feats = concat_interpolation(feats_a, feats_b, alpha)
            elif self.method == "mask":
                interp_feats = mask_interpolation(feats_a, feats_b, alpha)
            else:
                raise ValueError(f"Unknown interpolation method: {self.method}")
            # Normalize after interpolation if desired (some prefer it):
            interp_feats = F.normalize(interp_feats, p=2, dim=-1)
            result.append(interp_feats)
        return result


class SamplingPipeline:
    """
    A pipeline class for performing diffusion model sampling with CFG.
    """
    def __init__(self, diffusion_model: LightningDiT, sample_fn, latent_mean, latent_std, latent_multiplier):
        """
        Args:
            diffusion_model (LightningDiT): The diffusion model.
            sample_fn (callable): The sampling function, e.g. sampler or sample_fn in code.
            latent_mean (torch.Tensor): The mean latent vector from the training pipeline.
            latent_std (torch.Tensor): The std latent vector from the training pipeline.
            latent_multiplier (float): The multiplier used in pre-processing or post-processing latents.
        """
        self.model = diffusion_model
        self.sample_fn = sample_fn
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.latent_multiplier = latent_multiplier

    @torch.no_grad()
    def __call__(self, condition_vectors: torch.Tensor, cfg_scale: float = 7.75,
                 cfg_interval: bool = True, cfg_interval_start: float = 0.89,
                 n_images: int = 1, latent_size: int = 32, device: str = "cuda"):
        """
        Perform CFG sampling given condition vectors.

        Args:
            condition_vectors (torch.Tensor): Shape [B, feature_dim].
            cfg_scale (float): CFG scale.
            cfg_interval (bool): Whether to enable CFG interval trick.
            cfg_interval_start (float): The interval start for CFG.
            n_images (int): Number of images to generate per condition.
            latent_size (int): The latent spatial size for the diffusion model.
            device (str): The device to run on.

        Returns:
            torch.Tensor: The final latent after diffusion, shape [B*n_images, C, latent_size, latent_size].
        """
        bsz = condition_vectors.shape[0]
        # Repeat condition vectors (and the null condition) to perform CFG
        c_null = torch.zeros_like(condition_vectors, device=device)
        c_full = torch.cat([condition_vectors, c_null], dim=0)  # [2*B, feature_dim]

        # Prepare random noise z
        z = torch.randn(bsz, self.model.in_channels, latent_size, latent_size, device=device)
        # For CFG, we double the batch (condition vs. null)
        z = torch.cat([z, z], dim=0)  # [2 * B*n_images, C, H, W]

        # Sampling with CFG
        model_kwargs = dict(
            y=None,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            cfg_interval_start=cfg_interval_start,
            c=c_full
        )
        samples = self.sample_fn(z, self.model.forward_with_cfg, **model_kwargs)[-1]
        # Split the condition samples and null-condition samples
        condition_samples, _ = samples.chunk(2, dim=0)  # [B*n_images, C, H, W]

        # Convert latents from normalized form back to original
        # (x * latent_std) / latent_multiplier + latent_mean
        final_latents = (condition_samples * self.latent_std) / self.latent_multiplier + self.latent_mean
        return final_latents


class DecodingPipeline:
    """
    A pipeline class for decoding latents back to images using a VAE.
    """
    def __init__(self, vae_model):
        """
        Args:
            vae_model: The trained VAE model with a method decode_to_images().
        """
        self.vae_model = vae_model

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor):
        """
        Decode latents to images.

        Args:
            latents (torch.Tensor): shape [B, C, H, W].

        Returns:
            list[Image.Image]: A list of PIL images.
        """
        # vae_model.decode_to_images(...) returns a list of Tensors or np arrays
        decoded = self.vae_model.decode_to_images(latents)
        images = []
        for img_tensor in decoded:
            if isinstance(img_tensor, torch.Tensor):
                # Convert Tensor [3, H, W] to PIL
                img = Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            else:
                # If it's already a numpy array in HWC format
                img = Image.fromarray(img_tensor.astype(np.uint8))
            images.append(img)
        return images

