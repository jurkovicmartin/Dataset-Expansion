import cv2
import os

from utils import *


def split_images():
    """
    Splits images into left and right parts.
    """
    ### Settings

    IMG_DIR = "gdxray/split/img"
    MASK_DIR = "gdxray/split/mask"
    RESULTS_DIR = "gdxray/split2"

    padding = True
    pad_size = 30

    display_result = False

    result_img_dir = os.path.join(RESULTS_DIR, "img")
    result_mask_dir = os.path.join(RESULTS_DIR, "mask")

    if not os.path.exists(result_img_dir):
        os.makedirs(result_img_dir)

    if not os.path.exists(result_mask_dir):
        os.makedirs(result_mask_dir)

    ### Process

    images = load_images(IMG_DIR)
    masks = load_images(MASK_DIR)
    
    for i, (img, mask) in enumerate(zip(images, masks)):
        img_left, img_right = split_and_pad(img, padding, pad_size)
        # Masks without softening
        mask_left, mask_right = split_and_pad(mask, padding, pad_size, False)

        if display_result:
            display_images_vertically([img, img_left, img_right], [f"Image {i}", f"Left {i}", f"Right {i}"])
            display_images_vertically([mask, mask_left, mask_right], [f"Mask {i}", f"Left {i}", f"Right {i}"])

        cv2.imwrite(os.path.join(result_img_dir, f"img_left_{i}.png"), img_left)
        cv2.imwrite(os.path.join(result_img_dir, f"img_right_{i}.png"), img_right)
        cv2.imwrite(os.path.join(result_mask_dir, f"mask_left_{i}.png"), mask_left)
        cv2.imwrite(os.path.join(result_mask_dir, f"mask_right_{i}.png"), mask_right)

        

    



if __name__ == "__main__":
    split_images()
    
