import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE
from models.dinov2 import get_dino_v2_representation, get_dino_v2_model_256

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    """
    Run a tokenizer on full dataset and save the features.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup feature folders:
    output_dir = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.config))[0], f'{args.data_split}_{args.image_size}')
    if rank == 0:
        print(f"Saving features to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Create model:
    tokenizer = VA_VAE(
        args.config
    )
    
    encoder = get_dino_v2_model_256().to(device)

    # Setup data:
    datasets = [
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=0.0)),
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=1.0))
    ]
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        ) for dataset in datasets
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    representations = []
    representations_flip = []
    cls_tokens = []
    cls_tokens_flip = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images} of {total_data_in_loop} images')
        
        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]  # (N,)
            
            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)
            representation, cls_token = get_dino_v2_representation(raw_images=x.to(device), model=encoder)
            representation = representation.detach().cpu()
            cls_token = cls_token.detach().cpu()

            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            
            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
                representations.append(representation)
                cls_tokens.append(cls_token)
                
            else:
                latents_flip.append(z)
                representations_flip.append(representation)
                cls_tokens_flip.append(cls_token)

        if len(latents) == 10000 // args.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            representations = torch.cat(representations, dim=0)
            representations_flip = torch.cat(representations_flip, dim=0)
            cls_tokens = torch.cat(cls_tokens, dim=0)
            cls_tokens_flip = torch.cat(cls_tokens_flip, dim=0)
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'representations': representations,
                'representations_flip': representations_flip,
                'cls_tokens': cls_tokens,
                'cls_tokens_flip': cls_tokens_flip,
                'labels': labels
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            representations = []
            representations_flip = []
            cls_tokens = []
            cls_tokens_flip = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        representations = torch.cat(representations, dim=0)
        representations_flip = torch.cat(representations_flip, dim=0)
        cls_tokens = torch.cat(cls_tokens, dim=0)
        cls_tokens_flip = torch.cat(cls_tokens_flip, dim=0)
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'representations': representations,
            'representations_flip': representations_flip,
            'cls_tokens': cls_tokens,
            'cls_tokens_flip': cls_tokens_flip,
            'labels': labels
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    if rank == 0:
        dataset = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='ImageNet-1K/train')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="data/preprocessed/in1k256")
    parser.add_argument("--config", type=str, default="tokenizer/configs/vavae_f16d32.yaml")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()
    main(args)