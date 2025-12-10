import numpy as np
import os
from tqdm import tqdm
from PIL import Image # Use PIL to correctly read palettized PNGs

# --- Configuration ---
# The original PASCAL class IDs we are looking for
TARGET_CLASSES = {
    9,  # chair
    11, # diningtable
    15  # person
}

# Define the dataset lists we want to process
DATA_SPECS = [
    {
        'name': 'Training Set',
        'list_path': 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug_full.txt',
        'mask_dir': 'data/VOCdevkit/VOC2012/SegmentationClassAug',
        'output_path': 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_3class.txt'
    },
    {
        'name': 'Validation Set',
        'list_path': 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
        'mask_dir': 'data/VOCdevkit/VOC2012/SegmentationClass',
        'output_path': 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/val_3class.txt'
    }
]

def check_mask_for_targets(mask_path):
    """
    Checks if a given mask file contains any of our target class pixels.
    Returns True if a target is found, False otherwise.
    """
    try:
        mask_img = Image.open(mask_path)
    except FileNotFoundError:
        return False # File might be missing
    except Exception as e:
        print(f"Warning: Could not read {mask_path}, error: {e}")
        return False # File corrupt

    # Convert the PIL image to a numpy array
    mask = np.array(mask_img)
    
    # Find all unique pixel values in the mask
    unique_values = np.unique(mask)
    
    # Check if any of our target classes are in this list
    for value in unique_values:
        if value in TARGET_CLASSES:
            return True # Found a match!
            
    return False # No target classes found

# --- Main Script ---
def main():
    for spec in DATA_SPECS:
        print(f"--- Processing {spec['name']} ---")
        
        if not os.path.exists(spec['list_path']):
            print(f"Error: Cannot find source list file: {spec['list_path']}")
            continue

        with open(spec['list_path'], 'r') as f:
            all_files = [line.strip() for line in f if line.strip()]

        print(f"Found {len(all_files)} total images to check...")
        
        kept_files = []
        for file_id in tqdm(all_files, desc="Checking masks"):
            mask_path = os.path.join(spec['mask_dir'], file_id + '.png')
            
            if check_mask_for_targets(mask_path):
                kept_files.append(file_id)

        print(f"Finished. Kept {len(kept_files)} images (that contain our classes).")

        with open(spec['output_path'], 'w') as f:
            for file_id in kept_files:
                f.write(file_id + '\n')
        
        print(f"Saved new file list to: {spec['output_path']}\n")

if __name__ == '__main__':
    main()
