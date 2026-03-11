import os
import shutil
import logging

def combine_datasets(input_dirs: list[str], output_dir: str ="combined"):
    """
    Combines datasets where images and masks share the same name, renaming them sequentially to avoid overwriting across different source folders.
    Expected structure for each input directory:
        input_dir/
            img/
            mask/    
    Args:
        input_dirs (list[str]): List of directories to combine.
        output_dir (str, optional): Directory to save the combined dataset. Defaults to "combined".
    """
    out_img_dir = os.path.join(output_dir, "img")
    out_mask_dir = os.path.join(output_dir, "mask")
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    valid_extensions = (".png", ".jpg", ".jpeg")
    file_counter = 1

    ### Processing

    for in_dir in input_dirs:
        in_img_dir = os.path.join(in_dir, "img")
        in_mask_dir = os.path.join(in_dir, "mask")

        if not os.path.exists(in_img_dir) or not os.path.exists(in_mask_dir):
            print(f"Skipping '{in_dir}': Missing 'img' or 'mask' subfolder.")
            continue

        # Get all files in the current img folder (filter only images - valid extensions)
        img_files = sorted([f for f in os.listdir(in_img_dir) if os.path.isfile(os.path.join(in_img_dir, f)) and f.lower().endswith(valid_extensions)])

        for img_name in img_files:
            # Mask shares the same name
            mask_name = img_name 
            
            img_path = os.path.join(in_img_dir, img_name)
            mask_path = os.path.join(in_mask_dir, mask_name)

            # Ensure the matching mask actually exists
            if not os.path.exists(mask_path):
                print(f"WARNING: Mask '{mask_name}' not found in '{in_mask_dir}'. Skipping this pair.")
                continue

            # Extract the original file extensions
            img_ext = os.path.splitext(img_name)[1].lower()
            mask_ext = os.path.splitext(mask_name)[1].lower()

            # Create the new sequential file names (001, 002, etc.)
            new_img_name = f"{file_counter:03d}{img_ext}"
            new_mask_name = f"{file_counter:03d}{mask_ext}"

            img_dst = os.path.join(out_img_dir, new_img_name)
            mask_dst = os.path.join(out_mask_dir, new_mask_name)

            # Copy files to the unified directory
            shutil.copy2(img_path, img_dst)
            shutil.copy2(mask_path, mask_dst)

            file_counter += 1

        logging.info(f"{in_dir} processed.")

    logging.info(f"Successfully combined {file_counter - 1} image-mask pairs into '{output_dir}'.")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datasets = ["extended_set/flow", "extended_set/gdxray", "extended_set/inpaint", "extended_set/original"]
    
    output = "extended_set/combined"
    
    combine_datasets(datasets, output)