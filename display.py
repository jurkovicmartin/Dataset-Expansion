import cv2

from utils import *



def main(inpaint: bool =True):
    """
    False = Optical Flow

    Script only for displaying images/results.
    Not needed for the project functionality.
    """
    ### INPAINT
    if inpaint:
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
    ### OPTICAL FLOW
    else:
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



if __name__ == "__main__":
    # True = display inpainting
    # False = display optical flow
    main(False)

    # img = cv2.imread("data/img/77.png")
    # mask = cv2.imread("data/mask/77.png")
    # display_images([img, mask], ["Image", "Mask"])
