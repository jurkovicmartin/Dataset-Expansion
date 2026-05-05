import cv2
import numpy as np
import os
import logging
import math

from utils import load_images


class OpticalFlowManager:
    """
    Manager for optical flow. Works with individual images. They must be set manually (attributes - gt_img, gt_mask and target_img).
    Result is stored also as an attribute - propagated_mask.
    """
    def __init__(self, result_dir: str ="result/"):
        """
        Initializes the OpticalFlowManager object.

        Args:
            result_dir (str, optional): Directory where the propagated mask will be saved. Defaults to "result/".

        Attributes:
            gt_img (np.ndarray): Ground truth image.
            gt_mask (np.ndarray): Ground truth mask.
            target_img (np.ndarray): Target image to be propagated.
            propagated_mask (np.ndarray): Result of the propagation (target image with mask propagated).
        """
        self.result_dir = result_dir

        # Attributes critical for the functionality
        # ! Have to be set manually set first
        self.gt_img: np.ndarray = None
        self.gt_mask: np.ndarray = None
        self.target_img: np.ndarray = None
        # Output attribute
        self.propagated_mask: np.ndarray = None

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)


    def propagate_mask(self):
        """
        Propagates the mask from the ground truth image to the target image using optical flow (backward flow).

        Requires gt_img, gt_mask and target_img to be set first.
        """
        if self.gt_img is None or self.gt_mask is None or self.target_img is None:
            raise ValueError("GT image, mask and target image must be set first. (Attribute gt_img, gt_mask and target_img)")

        # Calculate backward flow
        # Flow from img to target_img
        flow = cv2.calcOpticalFlowFarneback(
            prev=self.target_img, 
            next=self.gt_img, 
            flow=None, 
            pyr_scale=0.5,     # Image scale to build pyramids (0.5 is standard)
            levels=3,          # Number of pyramid layers (increase if deformation is huge)
            winsize=15,        # Window size. Larger = handles faster changes, but blurs boundaries
            iterations=3,      # Number of iterations at each pyramid level
            poly_n=5,          # Pixel neighborhood to find polynomial expansion (5 or 7)
            poly_sigma=1.2,    # Standard deviation for the Gaussian (1.1 for 5, 1.5 for 7)
            flags=0
        )

        h, w = self.gt_img.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Add the flow vectors (dx, dy) to the grid coordinates
        map_x = np.float32(grid_x + flow[..., 0])
        map_y = np.float32(grid_y + flow[..., 1])

        self.propagated_mask = cv2.remap(
            self.gt_mask, 
            map_x, 
            map_y, 
            interpolation=cv2.INTER_NEAREST, # Keep integer values
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )


    def save_propagated_mask(self, filename: str ="propagated_mask.png"):
        """
        Saves the propagated mask to the result directory.

        Args:
            filename (str, optional): Name of the file to save. Defaults to "propagated_mask".

        Raises:
            ValueError: If the propagated mask is not calculated. (Call propagate_mask first)
        """
        if self.propagated_mask is None:
            raise ValueError("Propagated mask must be set first. (Call propagate_mask first)")
        
        cv2.imwrite(os.path.join(self.result_dir, f"{filename}"), self.propagated_mask)






from utils import *

def run():
    logging.basicConfig(level=logging.INFO)
    DIR = "flow/"

    samples = os.listdir(f"{DIR}")
    samples.sort()

    display_result = True

    manager = OpticalFlowManager()

    for sample in samples:
        # Load images
        images, images_names = load_images_with_names(f"{DIR}{sample}/img/")
        masks = load_images(f"{DIR}{sample}/mask/")
        target_images, target_names = load_images_with_names(f"{DIR}{sample}/target/")

        # Set output path
        result_path = f"{DIR}{sample}/result_2/"
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        manager.result_dir = result_path

        # How many targets per 1 GT ?
        target_ratio = math.ceil(len(target_images) / len(images))
        logging.info(f"Ratio for sample {sample}: {target_ratio} targets per GT")
        
        for target_idx in range(len(target_images)):
            img_idx = target_idx // target_ratio
            
            # Ensure img_idx doesn't exceed the list bounds 
            img_idx = min(img_idx, len(images) - 1)

            # Setting the attributes
            manager.gt_img = images[img_idx]
            manager.gt_mask = masks[img_idx]
            manager.target_img = target_images[target_idx]
            
            # Propagating
            manager.propagate_mask()
            propagated_mask = manager.propagated_mask

            manager.save_propagated_mask(f"propagated_mask_{target_names[target_idx]}")

            if display_result:
                display_images([images[img_idx], masks[img_idx], target_images[target_idx], propagated_mask],
                            [f"GT Image {images_names[img_idx]}",
                             f"GT Mask {images_names[img_idx]}",
                             f"Target Image {target_names[target_idx]}",
                             f"Propagated Mask {target_names[target_idx]}"])
                
                display_masks_comparison(masks[img_idx], propagated_mask,
                                         f"GT Mask {images_names[img_idx]}", f"Propagated Mask {target_names[target_idx]}")

        logging.info(f"Processed sample {sample}")
            



if __name__ == "__main__":
    run()
    