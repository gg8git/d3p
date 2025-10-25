import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import wandb

import logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG') == '1' else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)

def dummy_convert_batched_x_to_PIL_Image_list(self, x: torch.Tensor):
    raise Exception("x_to_PIL_Image is called before set by the setter method") 


class CustomSingletonLogger():
    _instance = None
    
    def __new__(cls, use_wandb: bool = False):
        run_name = os.getenv("RUN_NAME", "untitled_run")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.global_step = None
            cls._instance.convert_batched_x_to_PIL_Image_list = dummy_convert_batched_x_to_PIL_Image_list
            
            # Create timestamped run name
            from datetime import datetime
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            cls._instance._full_run_name = f"{run_name}_{timestamp}"
            
            if use_wandb:
                cls._instance.wandb_run = wandb.init(project="KitchenSink", name=cls._instance._full_run_name, entity="gg8-university-of-pennsylvania") # changed
                cls._instance.info("CustomSingletonLogger instantiated with W&B")
            else: 
                cls._instance.info("CustomSingletonLogger instantiated without W&B, calls to log_to_wandb_for_current_step will be ignored")
                cls._instance.wandb_run = None
        return cls._instance

    # NOTE (faraz): I am not sure why this requires stacklevel=3
    # based on documentaiton I would expect stacklevel=2 to be correct
    # there is something I don't understand here. 
    def debug(self, text: str):
        logging.debug(text, stacklevel=3)

    def info(self, text: str):
        logging.info(text, stacklevel=3)

    def warning(self, text: str):
        logging.warning(text, stacklevel=3)

    def error(self, text: str):
        logging.error(text, stacklevel=3)

    def log_to_wandb_for_current_step(self, items: dict):
        """
        This function wraps wandb.log so that the caller does not
        need to be aware of global_step. This prevents an extraneous 
        argument pass in the modeling code.
        """
        if self.wandb_run is not None:
            assert self.global_step is not None, "global_step has not yet been set by CustomSingletonLogger owner"
            items["global_step"] = self.global_step
            self.wandb_run.log(items)
        else: 
            self.warning("Called wandb log feature when wandb was never initialized.") 

    def log_config_to_wandb(self, config_dict: dict):
        """
        Log configuration parameters to wandb.
        """
        if self.wandb_run is not None:
            wandb.config.update(config_dict)
            self.info(f"Configuration logged to wandb: {config_dict}")
        else:
            self.warning("Called wandb config log feature when wandb was never initialized.")

    @property
    def run_name(self) -> str:
        return self._full_run_name 

    def set_global_step(self, step: int):
        self.global_step = step

    # This might be a shitty pattern but idrc atm it works
    def set_convert_batched_x_to_PIL_Image_list(self, foo: callable):
        self.convert_batched_x_to_PIL_Image_list = foo


## Visualization helper functions

def plot_imgs_as_gif(images: list[Image.Image], path: Path):

    gif = []
    for img in images:
        gif.append(img)

    gif[0].save(
        path,
        save_all=True,
        append_images=gif[1:],
        duration=100,
        loop=0,
    )

def plot_imgs_in_grid(imgs: list[Image.Image], path: Path = None):
    """Plot a list of PIL Images in a grid and optionally save to path. Returns the grid image."""
    
    # Calculate grid dimensions
    n_imgs = len(imgs)
    grid_size = int(n_imgs ** 0.5)
    if grid_size * grid_size < n_imgs:
        grid_size += 1
    
    # Get dimensions of first image
    img_width, img_height = imgs[0].size
    
    # Create new image for the grid
    grid_width = grid_size * img_width
    grid_height = grid_size * img_height
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images into grid
    for i, img in enumerate(imgs):
        row = i // grid_size
        col = i % grid_size
        x = col * img_width
        y = row * img_height
        grid_img.paste(img, (x, y))
    
    if path is not None:
        grid_img.save(path)
    
    return grid_img