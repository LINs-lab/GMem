# A toy code for constructing a memory bank using DINOv2 from custom images.
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from models.dinov2 import get_dino_v2_model_256, get_dino_v2_representation

# Data organization:
#   data_path/
#     folder1/
#       img1.png
#       img2.png
#       ...
#     folder2/
#       img3.png
#       img4.png
#       ...
#     ...

# Default hyper-parameters
DEFAULT_DATA_PATH = "data/preprocessed/in1kxxs256/images"  
DEFAULT_OUTP_PATH = "data/preprocessed/in1kxxs256/banks"  

class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from a directory structure.
    
    Transforms:
      1) Center-crop with aspect ratio preservation
      2) Convert to [0,1], then rescale to [-1,1]
    """
    def __init__(self, root_dir, image_size=256):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_paths = []
        
        # Recursively gather image paths
        for subfolder in os.listdir(root_dir):
            subdir = os.path.join(root_dir, subfolder)
            if os.path.isdir(subdir):
                for img_name in os.listdir(subdir):
                    full_path = os.path.join(subdir, img_name)
                    self.image_paths.append(full_path)
        
        self.image_paths = sorted(self.image_paths)
        
        # Define the transform pipeline
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self.center_crop_arr(img, image_size)),
            transforms.ToTensor(),
        ])
    
    @staticmethod
    def center_crop_arr(pil_image, image_size):
        """
        Center-crop preserving aspect ratio, following a multi-resolution approach:
          - Repeatedly downscale by factor of 2 until the smallest side is < 2 * image_size.
          - Scale to make the smallest side = image_size.
          - Finally, center-crop to (image_size, image_size).
        """
        while min(pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                (pil_image.size[0] // 2, pil_image.size[1] // 2),
                resample=Image.BOX
            )
        
        scale = image_size / min(pil_image.size)
        new_width = round(pil_image.size[0] * scale)
        new_height = round(pil_image.size[1] * scale)
        pil_image = pil_image.resize((new_width, new_height), resample=Image.BICUBIC)
        
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        arr = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
        
        return Image.fromarray(arr)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and convert to RGB
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        image_tensor = self.transform(image)
        # Rescale from [0,1] to [-1,1], for compatibility with dinov2.py
        image_tensor = 2.0 * image_tensor - 1.0
        
        return (image_tensor, )

def main(args):
    """
    Main function to:
      1) Load a dataset from `args.data_path`.
      2) Pass images through DINOv2 to extract snippets.
      3) Save all snippets into a memory bank at `args.output_path`.
    """
    
    # Resolve data path
    data_path = args.data_path if args.data_path else DEFAULT_DATA_PATH
    if not data_path:
        raise ValueError("No valid data_path provided, and DEFAULT_DATA_PATH is empty.")
    
    # Resolve output path
    output_path = args.output_path if args.output_path else DEFAULT_OUTP_PATH
    if not output_path:
        raise ValueError("No valid output_path provided, and DEFAULT_OUTP_PATH is empty.")
    
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    
    
    dataset = CustomImageDataset(root_dir=data_path, image_size=256)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the DINOv2 model
    print("Loading DINOv2 model...")
    encoder = get_dino_v2_model_256().to(device)
    encoder.eval()  # inference mode
    
    
    # We'll collect snippets batch by batch
    memory_bank_list = []
    
    print("Extracting snippets to build the memory bank...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Building Memory Bank"):
            images = batch[0].to(device, non_blocking=True)
            
            # Extract memory snippet, refer to models.dinov2
            _, snippets = get_dino_v2_representation(images, encoder)
            
            memory_bank_list.append(snippets.cpu())
            
    memory_bank = torch.cat(memory_bank_list, dim=0)
    
    
    os.makedirs(output_path, exist_ok=True)
    memory_bank_filename = f"{args.dataset}_memory_bank.pth" if args.dataset else "memory_bank.pth"
    memory_bank_path = os.path.join(output_path, memory_bank_filename)
    torch.save(memory_bank, memory_bank_path)
    print(f"Memory bank saved to: {memory_bank_path}")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Memory Bank Construction with DINOv2")
    parser.add_argument("--data-path", type=str, default="",
                        help="Path to the directory containing images. If empty, falls back to DEFAULT_DATA_PATH.")
    parser.add_argument("--output-path", type=str, default="",
                        help="Path to the output directory. If empty, falls back to DEFAULT_OUTPUT_PATH.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for the DataLoader.")
    parser.add_argument("--dataset", type=str,
                        help="Optional dataset name for naming the output memory bank file.")
    
    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
