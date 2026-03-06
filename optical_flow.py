import cv2
import numpy as np
import os
import logging

from utils import load_images


class OpticalFlowManager:
    """
    Manager for optical flow. Works with individual images. They must be set manually (attributes - gt_img, gt_mask and next_img).
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
            next_img (np.ndarray): Next image to be propagated.
            propagated_mask (np.ndarray): Result of the propagation (next image with mask propagated).
        """
        self.result_dir = result_dir

        # Attributes critical for the functionality
        # ! Have to be set manually set first
        self.gt_img: np.ndarray = None
        self.gt_mask: np.ndarray = None
        self.next_img: np.ndarray = None
        # Output attribute
        self.propagated_mask: np.ndarray = None

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)


    def propagate_mask(self):
        """
        Propagates the mask from the ground truth image to the next image using optical flow (backward flow).

        Requires gt_img, gt_mask and next_img to be set first.
        """
        if self.gt_img is None or self.gt_mask is None or self.next_img is None:
            raise ValueError("GT image, mask and next image must be set first. (Attribute gt_img, gt_mask and next_img)")

        # Calculate backward flow
        # Flow from img to next_img
        flow = cv2.calcOpticalFlowFarneback(
            prev=self.next_img, 
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
        images = load_images(f"{DIR}{sample}/img/")
        masks = load_images(f"{DIR}{sample}/mask/")
        next_images = load_images(f"{DIR}{sample}/next/")

        # Set output path
        result_path = f"{DIR}{sample}/result/"
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        manager.result_dir = result_path

        for i in range(len(images)):
            # Setting the attributes
            manager.gt_img = images[i]
            manager.gt_mask = masks[i]
            manager.next_img = next_images[i]

            # Propagating
            manager.propagate_mask()
            propagated_mask = manager.propagated_mask

            manager.save_propagated_mask(f"propagated_mask_{i}.png")

            if display_result:
                display_images([images[i], masks[i], next_images[i], propagated_mask],
                            [f"GT Image {i}", f"GT Mask {i}", f"Image {i}", f"Propagated Mask {i}"])
                
                display_masks_comparison(masks[i], propagated_mask, f"GT Mask {i}", f"Propagated Mask {i}")

        logging.info(f"Processed sample {sample}")



if __name__ == "__main__":
    run()
    