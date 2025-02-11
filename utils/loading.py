from transport import create_transport, Sampler
from tokenizer.vavae import VA_VAE
import torch
import os
import yaml
from models.lightningdit import LightningDiT_models, LightningDiT
from utils.svd_utils import bank_models, BaseMemoryBank


def get_model(train_config:dict) -> LightningDiT:
    
    ckpt_dir = train_config['ckpt_path']

    if 'downsample_ratio' in train_config['vae']:
        latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    else:
        latent_size = train_config['data']['image_size'] // 16

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
    return model

def get_output_dir(train_config: dict):
    """
    Get output directory from config.
    """
    output_dir = train_config['train']['output_dir']
    exp_name = train_config['train']['exp_name']
    demo_dir = 'demo_images'
    
    final_output_dir = os.path.join(output_dir, exp_name, demo_dir)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    return final_output_dir


def load_vae(train_config: dict, vae=None):
    """
    Load vae from checkpoint
    """
    
    if vae is None:
        vae = VA_VAE(
            f'tokenizer/configs/{train_config["vae"]["model_name"]}.yaml',
        )
    
    return vae


def load_bank(train_config: dict, device='cpu'):
    """
    Load bank from config
    """
    bank_type = train_config['GMem']['bank_type']
    bank_path = train_config['GMem']['bank_path']
    bank: BaseMemoryBank = bank_models[bank_type].from_state_dict(bank_path).to(device)
    
    return bank

    
def get_transport(train_config: dict, transport=None):
    """
    Load transport from config
    """
    if transport is None:
        transport = create_transport(
            train_config['transport']['path_type'],
            train_config['transport']['prediction'],
            train_config['transport']['loss_weight'],
            train_config['transport']['train_eps'],
            train_config['transport']['sample_eps'],
            use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
            use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
        )  # default: velocity;
    
    return transport
    
def get_sampler(transport, sampler=None):
    """
    Load sampler from config
    """
    if sampler is None:
        sampler = Sampler(transport)
    
    return sampler

def get_sample_fn(train_config: dict, sampler, timestep_shift):
    """
    Load sample_fn from config
    """
    mode = train_config['sample']['mode']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=timestep_shift,
        )
    else:
        sample_fn = sampler.sample_sde(
            sampling_method=train_config['sample']['sampling_method'],
            diffusion_form=train_config['sample']['diffusion_form'],
            diffusion_norm=train_config['sample']['diffusion_norm'],
            last_step=train_config['sample']['last_step'],
            last_step_size=train_config['sample']['last_step_size'],
            num_steps=train_config['sample']['num_sampling_steps'],
        )
    return sample_fn

def get_latents(train_config: dict, device='cpu'):
    from utils.data_utils import get_latent_stats_from_local
    latent_mean, latent_std = get_latent_stats_from_local(path='pretrains/vavae/latents_stats.pt')
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    return latent_mean.to(device), latent_std.to(device), latent_multiplier

def get_latent_size(train_config: dict):
    if 'downsample_ratio' in train_config['vae']:
        latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    else:
        latent_size = train_config['data']['image_size'] // 16
    return latent_size

def get_necessary_data(train_config: dict):
    """
    Load necessary data from config
    """
    timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0
    latent_size = get_latent_size(train_config)
    transport = get_transport(train_config)
    sampler = get_sampler(transport)
    sample_fn = get_sample_fn(train_config, sampler, timestep_shift)
    vae = load_vae(train_config)
    latent_mean, latent_std, latent_multiplier = get_latents(train_config)
    output_dir = get_output_dir(train_config)
    
    return timestep_shift, latent_size, transport, sampler, sample_fn, vae, latent_mean, latent_std, latent_multiplier, output_dir


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
