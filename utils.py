import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
import cv2
import os


############################### DISPLAYING #######################################################

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
    diff_img = np.full((*mask_a_bool.shape, 3), 220, dtype=np.uint8)
    INTERSECTION_COLOR = [0, 0, 139]
    A_COLOR = [255, 0, 255]
    B_COLOR = [0, 255, 0]  
    diff_img[mask_a_bool & mask_b_bool] =  INTERSECTION_COLOR
    diff_img[mask_a_bool & ~mask_b_bool] = A_COLOR
    diff_img[~mask_a_bool & mask_b_bool] = B_COLOR
    
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
    patch_a = mpatches.Patch(color=[c/255 for c in A_COLOR], label=f"Only in {title_a}")
    patch_b = mpatches.Patch(color=[c/255 for c in B_COLOR], label=f"Only in {title_b}")
    patch_inter = mpatches.Patch(color=[c/255 for c in INTERSECTION_COLOR], label="Intersection")
    
    axes[2].legend(handles=[patch_a, patch_b, patch_inter], 
                  loc="lower center", bbox_to_anchor=(0.5, -0.2), 
                  ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()


def display_images_vertically(images: list[np.ndarray], titles: list[str] =None):
    """
    Displays a set of images in a vertical layout.

    Args:
        images (list[np.ndarray]): List of images to display.
        titles (list[str], optional): List of titles to image. Defaults to None.

    Raises:
        ValueError: If the number of images and titles is not the same.
    """
    num_images = len(images)

    if titles:
        if num_images != len(titles):
            raise ValueError("Number of images and titles must be the same.")
    
    # Scale the height based on the number of images
    fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(8, 5 * num_images))

    # Only 1 image
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        axes[i].imshow(images[i], cmap="gray")

        if titles:
            axes[i].set_title(titles[i])
        else:
            axes[i].set_title(f"Image {i}")
        
        axes[i].axis("off")        

    plt.tight_layout()
    plt.show()



################################################### IMAGES ############################################################

def load_images(dir_path: str) -> list[np.ndarray]:
    """
    Loads all images in a directory.

    Args:
        dir_path (str): Directory path containing the images to load.

    Returns:
        list[np.ndarray]: A list of grayscale images.
    """
    output = []

    names = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    names.sort()

    for name in names:
        path = os.path.join(dir_path, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        output.append(img)

    return output


def load_images_with_names(dir_path: str) -> tuple[list[np.ndarray], list[str]]:
    """
    Loads all images in a directory and returns also their names.

    Args:
        dir_path (str): Directory path containing the images to load.

    Returns:
        tuple[list[np.ndarray], list[str]]: A tuple containing a list of grayscale images and a list of their corresponding names.
    """
    output = []

    names = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    names.sort()

    for name in names:
        path = os.path.join(dir_path, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        output.append(img)

    return output, names


def resize_and_pad(img: np.ndarray, size: int =256) -> np.ndarray:
        """
        Resizes the given image to the given size, and pads it to fit the size if necessary.

        Args:
            img (np.ndarray): The image to be resized and padded.
            size (int, optional): The size to resize the image to. Defaults to 256.

        Returns:
            np.ndarray: The resized and padded image.

        Raises:
            ValueError: If the given image is not a 2D array.
        """
        height, width = img.shape
        scale = size / max(height, width)
        new_w, new_h = int(width * scale), int(height * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        new_img = np.full((size, size), 0, dtype=img.dtype)
        
        # Center the resized image
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        new_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return new_img


def split_and_pad(img: np.ndarray, padding=False, pad_width=30, soften=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the given image into two halves (left and right) and optionally adds black padding to the sides of the image and softens the edges.

    Args:
        img (np.ndarray): The image to be split.
        padding (bool, optional): Whether to add black padding to the sides of the image. Defaults to False.
        pad_width (int, optional): The width of the padding. Defaults to 30.
        soften (bool, optional): Whether to soften the edges of the image using a Gaussian Blur. Softening is only applied if there is also padding. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: The two halves of the image.
    """
    h, w = img.shape
    midpoint = w // 2
    left_half = img[:, :midpoint]
    right_half = img[:, midpoint:]

    if padding:
        left_half = pad_and_soften(left_half, pad_width, soften)
        right_half = pad_and_soften(right_half, pad_width, soften)

    return left_half, right_half



def pad_and_soften(img: np.ndarray, pad_width: int =30, soften: bool =True) -> np.ndarray:
    """
    Pads the given image with a black border and optionally softens the edges of the image.

    Args:
        img (np.ndarray): The image to be padded and softened.
        pad_width (int, optional): The width of the padding. Defaults to 30.
        soften (bool, optional): Whether to soften the edges of the image. Defaults to True.

    Returns:
        np.ndarray: The padded and softened image.
    """
    ### PADDING

    h, w = img.shape
    # Larger black canvas
    canvas_h, canvas_w = h + 2*pad_width, w + 2*pad_width
    padded_img = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Place the image in the middle of canvas
    padded_img[pad_width:pad_width + h, pad_width:pad_width + w] = img

    if not soften:
        return padded_img

    ### SOFTENING

    mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    mask[pad_width:pad_width+h, pad_width:pad_width+w] = 1.0

    # Blur the mask to create the "feather" effect
    blur_strength = 31
    mask = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)
    
    # Apply the mask
    soft_edged = (padded_img.astype(float) * mask).astype(np.uint8)

    return soft_edged
