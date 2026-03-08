"""
Script only for displaying -- does not affect project's functionality.

Displaying settings are done within each display block.
"""
import cv2

from utils import *



def main(display: str ="original"):
    """
    Displays images based on the given display type.

    Args:
        display (str): Display type. Can be "original", "inpaint", "flow", or "gdxray".
    """
    if display == "original":
        slice = "77"

        img_path = f"data/img/{slice}.png"
        mask_path = f"data/mask/{slice}.png"

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        display_images([img, mask], [f"Image {slice}", f"Mask {slice}"])

    elif display == "inpaint":
        to_display = ["14", "15", "35", "41"]
        # to_display = ["01"]

        for slice in to_display:

            img_path = f"inpaint/img/{slice}.png"
            mask_path = f"inpaint/mask/{slice}.png"
            inpainted_path = f"inpaint/result/{slice}.png"

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            inpainted = cv2.imread(inpainted_path, cv2.IMREAD_GRAYSCALE)

            display_images([img, mask, inpainted], [f"Image {slice}", f"Mask {slice}", f"Inpainted {slice}"])

            display_overlays([img, img, inpainted, inpainted],
                            [np.zeros_like(img), mask, np.zeros_like(inpainted), mask],
                            [f"Image {slice}", f"Image {slice}", f"Inpainted {slice}", f"Inpainted {slice}"])
    
    elif display == "flow":
        sample = "12"
        slice = "225"
        next_slice = "230"
        slice_position = "0"

        img_path = f"flow/{sample}/img/{slice}.png"
        mask_path = f"flow/{sample}/mask/{slice}.png"
        next_img_path = f"flow/{sample}/next/{next_slice}.png"
        next_mask_path = f"flow/{sample}/result/propagated_mask_{slice_position}.png"

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_img_path, cv2.IMREAD_GRAYSCALE)
        next_mask = cv2.imread(next_mask_path, cv2.IMREAD_GRAYSCALE)

        display_images([img, mask, next_img, next_mask],
                       [f"Image {sample}/{slice}", f"Mask {sample}/{slice}",
                        f"Next Image {sample}/{next_slice}", f"Next Mask {sample}/{next_slice}"])

        display_overlays([img, img, next_img, next_img],
                        [mask, np.zeros_like(img), next_mask, np.zeros_like(next_img)],
                        [f"Image {sample}/{slice}", f"Image {sample}/{slice}",
                         f"Next Image {sample}/{next_slice}", f"Next Image {sample}/{next_slice}"])
        
        display_masks_comparison(mask, next_mask, f"Mask {sample}/{slice}", f"Next Mask {sample}/{next_slice}")

    elif display == "gdxray":
        sample = "01"
        splitted_index = "0"

        img_path = f"gdxray/img/{sample}.png"
        mask_path = f"gdxray/mask/{sample}.png"

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        left_img = cv2.imread(f"gdxray/splitted/img/img_left_{splitted_index}.png", cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(f"gdxray/splitted/img/img_right_{splitted_index}.png", cv2.IMREAD_GRAYSCALE)
        left_mask = cv2.imread(f"gdxray/splitted/mask/mask_left_{splitted_index}.png", cv2.IMREAD_GRAYSCALE)
        right_mask = cv2.imread(f"gdxray/splitted/mask/mask_right_{splitted_index}.png", cv2.IMREAD_GRAYSCALE)

        display_images_vertically([img, mask], [f"Image {sample}", f"Mask {sample}"])

        display_images([left_img, right_img, left_mask, right_mask],
                       [f"Left Image {sample}", f"Right Image {sample}", f"Left Mask {sample}", f"Right Mask {sample}"])

    else:
        raise ValueError(f"Unknown display type: {display}")        



if __name__ == "__main__":
    # Setting for each display (paths) are set within each display block
    main(display="gdxray")

    


