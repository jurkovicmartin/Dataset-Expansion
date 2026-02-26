import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
import cv2


def display_images(images: list[np.ndarray], titles: list[str] =None):
    """
    Displays a set of images in a grid layout.

    Args:
        images (list[np.ndarray]): List of images to display.
        titles (list[str], optional): List of titles to image. Defaults to None.

    Raises:
        ValueError: If the number of images and titles is not the same.
    """
    n = len(images)
    cols = 2
    rows = math.ceil(n / cols)

    if titles:
        if n != len(titles):
            raise ValueError("Number of images and titles must be the same.")

    plt.figure(figsize=(10, 4 * rows))
    
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        
        plt.imshow(img, cmap="gray")
        
        if titles:
            plt.title(titles[i])
        else:
            plt.title(f"Image {i}")
            
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def display_overlays(images: list[np.ndarray], masks: list[np.ndarray], titles: list[str] =None):
    """
    Displays a set of images with overlays in a grid layout.

    Args:
        images (list[np.ndarray]): List of images to display.
        masks (list[np.ndarray]): List of masks to overlay on the images.
        titles (list[str], optional): List of titles to image. Defaults to None.

    Raises:
        ValueError: If the number of images and titles, or images and masks is not the same.
    """
    n = len(images)
    cols = 2
    rows = math.ceil(n / cols)

    if titles:
        if n != len(titles):
            raise ValueError("Number of images and titles must be the same.")
        
    if n != len(masks):
        raise ValueError("Number of images and masks must be the same.")
    
    # Define the overlay parameters
    alpha = 0.5
    color = (255, 0, 0)

    plt.figure(figsize=(12, 5 * rows))
    
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        
        # Convert grayscale image to 3-channel RGB
        bg_color = cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB)
        
        # Create a colored canvas the same size as the image
        overlay = np.zeros_like(bg_color)
        overlay[:] = color
        
        # Create the blended image
        output = bg_color.copy()
        mask_indices = masks[i] > 0
        
        # Blend the background image with the overlay, using the mask
        output[mask_indices] = cv2.addWeighted(bg_color, 1 - alpha, overlay, alpha, 0)[mask_indices]
        plt.imshow(output)

        if titles:
            plt.title(titles[i])
        else:
            plt.title(f"Overlay {i}")

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def display_masks_comparison(mask_a: np.ndarray, mask_b: np.ndarray, title_a="Mask A", title_b="Mask B"):
    """
    Displays a comparison of two masks by creating a difference map.

    Args:
        mask_a (np.ndarray): The first mask to compare.
        mask_b (np.ndarray): The second mask to compare.
        title_a (str, optional): The title of mask A. Defaults to "Mask A".
        title_b (str, optional): The title of mask B. Defaults to "Mask B".
    """
    # Boolean for logical operations
    mask_a_bool = mask_a.astype(bool)
    mask_b_bool = mask_b.astype(bool)

    # Create  difference map
    diff_img = np.zeros((*mask_a_bool.shape, 3), dtype=np.uint8)
    diff_img[mask_a_bool & mask_b_bool] = [0, 0, 255] # Intersection
    diff_img[mask_a_bool & ~mask_b_bool] = [255, 0, 0]    # Unique to A
    diff_img[~mask_a_bool & mask_b_bool] = [0, 255, 0]    # Unique to B
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Mask A
    axes[0].imshow(mask_a, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(title_a)
    axes[0].axis('off')
    
    # Mask B
    axes[1].imshow(mask_b, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(title_b)
    axes[1].axis('off')
    
    # Difference
    axes[2].imshow(diff_img)
    axes[2].set_title("Difference Map")
    axes[2].axis('off')
    
    # Legend
    red_patch = mpatches.Patch(color="red", label=f"Only in {title_a}")
    green_patch = mpatches.Patch(color="lime", label=f"Only in {title_b}")
    white_patch = mpatches.Patch(color="blue", label="Intersection")
    
    axes[2].legend(handles=[red_patch, green_patch, white_patch], 
                  loc="lower center", bbox_to_anchor=(0.5, -0.2), 
                  ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()

