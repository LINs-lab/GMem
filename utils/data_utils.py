import os
import torch
from math import ceil
from typing import List
from PIL import Image
import numpy as np
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_latent_stats_from_local(path: str):
    """
    Get the latent statistics from the local path
    """
    assert os.path.exists(path), f"Path {path} does not exist"
    latent_stats = torch.load(path)
    return latent_stats['mean'], latent_stats['std']

from typing import List
def load_images_from_local(path: List[str]):
    """
    Get the images from the local path, supporting various image formats.
    """
    import pathlib
    from PIL import Image
    
    images = []  
    
    for image_path in path:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0) 
        images.append(image)
    
    return torch.cat(images, dim=0) if images else torch.Tensor()



def create_image_grid(all_images: List[torch.Tensor], img_w: int, img_h: int, n_cols: int) -> Image:
    """
    Creates a grid image from a list of images.
    
    Args:
        all_images (list of PIL.Image): List of images to arrange in a grid.
        img_w (int): The desired width for each image cell.
        img_h (int): The desired height for each image cell.
        n_cols (int): The number of columns in the grid.
        
    Returns:
        PIL.Image: A new image containing all input images arranged in a grid.
    
    Raises:
        ValueError: If all_images is empty or n_cols is less than 1.
    """
    
    if not all_images:
        raise ValueError("The list 'all_images' is empty.")
    if n_cols < 1:
        raise ValueError("n_cols must be at least 1.")
    
    # Ensure all images are resized to the desired dimensions
    resized_images = []
    for tensor in all_images:
        if type(tensor) is np.ndarray:
            pil_img = Image.fromarray(tensor.astype(np.uint8))
            resized_images.append(pil_img)
        else:
            resized_images.append(tensor)
    
    # Determine the grid size
    n_images = len(resized_images)
    n_rows = ceil(n_images / n_cols)
    
    # Create a new blank image to hold the grid
    grid_width = n_cols * img_w
    grid_height = n_rows * img_h
    grid_img = Image.new("RGB", (grid_width, grid_height))
    
    # Paste each image into the grid
    for idx, img in enumerate(resized_images):
        row = idx // n_cols
        col = idx % n_cols
        x = col * img_w
        y = row * img_h
        grid_img.paste(img, (x, y))
    
    return grid_img
