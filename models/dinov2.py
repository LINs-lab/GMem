import torch
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

def image_from_minus_one_to_one(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.) / 2.


def preprocess_raw_image(x: torch.Tensor) -> torch.Tensor:
    resolution = x.shape[-1]
    x = image_from_minus_one_to_one(x)
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x

def get_dino_v2_model_256():
    import timm

    encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
    del encoder.head
    patch_resolution = 16 * (256 // 256)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data, [patch_resolution, patch_resolution],
    )
    encoder.head = torch.nn.Identity()
    encoder.eval()
    
    return encoder

@torch.no_grad()
def get_dino_v2_representation(raw_images: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    raw_image_ = preprocess_raw_image(raw_images)
    z = model.forward_features(raw_image_)
    cls_token = z['x_norm_clstoken']
    x_norm_patchtokens = z['x_norm_patchtokens']
        
    return x_norm_patchtokens, cls_token
        
        
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_dino_v2_model_256().to(device)
    print(model)
    x = torch.randn(10, 3, 256, 256).to(device)
    z = get_dino_v2_representation(x, model) # (10, 256, 768)
    print(z.shape)