import argparse, torch, numpy as np
from PIL import Image
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel as DDP
from models.lightningdit import LightningDiT_models
import torch.nn.functional as F
from utils.logger import print_with_prefix
from utils.loading import load_bank, get_necessary_data, load_config
    
    
# sample function
@torch.no_grad()
def do_demo_sample(train_config, accelerator, model, train_steps, bank=None):
    """
    Run sampling.
    """
    if accelerator.process_index == 0:

        assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"

        # Setup accelerator:
        model.eval()
        
        model = model.module if isinstance(model, DDP) else model
        
        device = accelerator.device
        
        if bank is None:
            bank = load_bank(train_config, device=device)

        timestep_shift, latent_size, transport, sampler, sample_fn, vae, latent_mean, latent_std, latent_multiplier, output_dir = get_necessary_data(train_config)
                
        model, latent_mean, latent_std, bank, vae.model = map(lambda x: x.to(device), (model, latent_mean, latent_std, bank, vae.model))
            
        images = []
        for label in [781545, 1100424, 1254779, 306839, 842428, 565858, 303966, 317503]:
            z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
            y = None
            c = F.normalize(bank[label], p=2, dim=-1).unsqueeze(0)
            model_kwargs = dict(y=y, c=c)
            model_fn = model.forward
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            samples = (samples * latent_std) / latent_multiplier + latent_mean
            samples = vae.decode_to_images(samples)
            images.append(samples)
        all_images = np.stack([img[0] for img in images])  # Take first image from each batch            
        h, w = all_images.shape[1:3]
        grid = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
        for idx, image in enumerate(all_images):
            i, j = divmod(idx, 4)  # Calculate position in 2x4 grid
            grid[i*h:(i+1)*h, j*w:(j+1)*w] = image
            
        # Save the combined image
        Image.fromarray(grid).save(f'{output_dir}/{train_steps}.png')
        
        print(f"Demo image of {train_steps} steps saved to {output_dir}/{train_steps}.png")
        del images, all_images, grid, latent_mean, latent_std, latent_multiplier, device, sample_fn, transport, sampler, timestep_shift, latent_size
    
        model.train()
    
    accelerator.wait_for_everyone()
    return None
        

        

# some utils

if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lightningdit_b_ldmvae_f16d16.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()
    accelerator = Accelerator()
    train_config = load_config(args.config)

    # get ckpt_dir
    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    if accelerator.process_index == 0:
        print_with_prefix('Using ckpt:', train_config['ckpt_path'])
    ckpt_dir = train_config['ckpt_path']

    if 'downsample_ratio' in train_config['vae']:
        latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    else:
        latent_size = train_config['data']['image_size'] // 16

    # get model
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        learn_sigma=train_config['model']['learn_sigma'] if 'learn_sigma' in train_config['model'] else False,
    )

    checkpoint = torch.load(ckpt_dir, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()  # important!
        
    # naive sample
    sample_folder_dir = do_demo_sample(train_config, accelerator, model=model, bank=None, train_steps=200000)
    print_with_prefix('Sampling Done!')
        
    accelerator.wait_for_everyone()
    accelerator.end_training()
    