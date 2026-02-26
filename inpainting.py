import cv2
import numpy as np
import os


class InpaintingManager:
    """
    Manager for inpainting images using OpenCV. Works with directories.
    """
    def __init__(self, img_dir: str, mask_dir: str, result_dir: str ="result/", inpaint_radius: int =3):
        """
        Initialize the InpaintingManager object.

        Args:
            img_dir (str): Directory containing the images to be inpainted.
            mask_dir (str): Directory containing the masks of the images.
            result_dir (str, optional): Directory where the inpainted images will be saved. Defaults to "result/".
            inpaint_radius (int, optional): Radius of the inpainting algorithm. Defaults to 3.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.result_dir = result_dir
        self.inpaint_radius = inpaint_radius

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Load image and mask names
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.img_names.sort()
        self.mask_names = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
        self.mask_names.sort()

        if len(self.img_names) != len(self.mask_names):
            raise ValueError("Number of images and masks must be the same.")


    def load_image_and_mask(self, index: int =0) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads an image and its corresponding mask. Indexes correspond to the position of the image and mask in the directories.

        Args:
            index (int, optional): Index of the image and mask to load. Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the image and its mask.
        """
        image_path = os.path.join(self.img_dir, self.img_names[index])
        mask_path = os.path.join(self.mask_dir, self.mask_names[index])
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Ensure the mask is strictly 0 and 255
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return image, mask
    

    def inpaint(self, img, mask) -> np.ndarray:
        """
        Inpaints an image based on a given mask.

        Args:
            img (np.ndarray): The image to be inpainted.
            mask (np.ndarray): The mask indicating the region to be inpainted.

        Returns:
            np.ndarray: The inpainted image.
        """
        # Set inpainting algorithm
        method = cv2.INPAINT_NS

        # Dilate the mask slightly to capture the noise around the defects
        kernel = np.ones((10, 10), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # display_images([img, mask, dilated_mask], ["Image", "Mask", "Dilated Mask"])

        return cv2.inpaint(img, dilated_mask, inpaintRadius=self.inpaint_radius, flags=method)
    

    def inpaint_full_set(self):
        """
        Inpaints all images in the dataset (directory).

        Iterates over all image-mask pairs in the dataset, inpaints the image, and saves the result in the result directory.
        """
        for i, name in enumerate(self.img_names):
            image, mask = self.load_image_and_mask(i)
            synthetic = self.inpaint(image, mask)

            cv2.imwrite(os.path.join(self.result_dir, f"{name}"), synthetic)






from utils import *


def run():
    IMAGES_DIR = "data/img"
    MASKS_DIR = "data/mask"
    RESULTS_DIR = "result/inpaint/"

    display_result = False

    manager = InpaintingManager(IMAGES_DIR, MASKS_DIR, RESULTS_DIR, inpaint_radius=5)
    manager.inpaint_full_set()

    if not display_result:
        return
    
    file_names = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")]
    file_names.sort()

    for name in file_names:
        img = cv2.imread(f"{IMAGES_DIR}/{name}", cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(f"{MASKS_DIR}/{name}", cv2.IMREAD_GRAYSCALE)
        synthetic = cv2.imread(f"{RESULTS_DIR}/{name}", cv2.IMREAD_GRAYSCALE)

        display_images([img, mask, synthetic], [f"Image {name}", f"Mask {name}", f"Synthetic {name}"])

        display_overlays([img, img, synthetic, synthetic],
                         [mask, np.zeros_like(img), mask, np.zeros_like(synthetic)],
                         [f"Image {name}", f"Image {name}", f"Synthetic {name}", f"Synthetic {name}"])


    # Partial inpainting of the set
    # For testing
    # for i in range(10):
    #     img, mask = manager.load_image_and_mask(i)
    #     synthetic = manager.inpaint(img, mask)

    #     display_images([img, mask, synthetic], [f"Image {i}", f"Mask {i}", f"Synthetic {i}"])

    #     display_overlays([img, img, synthetic, synthetic],
    #                      [mask, np.zeros_like(img), mask, np.zeros_like(synthetic)],
    #                      [f"Image {i}", f"Image {i}", f"Synthetic {i}", f"Synthetic {i}"])


if __name__ == "__main__":
    run()

