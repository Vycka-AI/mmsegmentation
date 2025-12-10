import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- Config ---
# Source folders (Standard PASCAL locations)
SRC_TRAIN_DIR = 'data/VOCdevkit/VOC2012/SegmentationClass'
SRC_VAL_DIR = 'data/VOCdevkit/VOC2012/SegmentationClass'

# Destination folders (New 4-class masks)
DST_TRAIN_DIR = 'data/VOCdevkit/VOC2012/SegmentationClass4Class_Train'
DST_VAL_DIR = 'data/VOCdevkit/VOC2012/SegmentationClass4Class_Val'

# Use the ORIGINAL PASCAL lists (No filtering)
TRAIN_LIST = 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
VAL_LIST = 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

# The Mapping
# 0 (Background) -> 0 (Background)
# 15 (Person)    -> 1 (Person)
# 9 (Chair)      -> 2 (Chair)
# 11 (Table)     -> 3 (Table)
MAPPING = {
    0: 0,
    15: 1,
    9: 2,
    11: 3
    # All others default to 255 (Ignore)
}

def convert_folder(list_file, src_dir, dst_dir):
    print(f"Processing list: {list_file}")
    
    if not os.path.exists(list_file):
        print(f"Error: List file not found: {list_file}")
        return

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with open(list_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]

    print(f"Converting {len(file_names)} masks...")

    for name in tqdm(file_names):
        src_path = os.path.join(src_dir, name + '.png')
        dst_path = os.path.join(dst_dir, name + '.png')

        try:
            # Open using PIL to preserve palette indices
            mask = Image.open(src_path)
            mask_arr = np.array(mask)
        except FileNotFoundError:
            # Some images in the list might not have segmentation masks in standard VOC
            # This is normal for the 'train.txt' vs 'trainval.txt' distinction
            continue 

        # Create new mask filled with 255 (Ignore)
        new_mask = np.full_like(mask_arr, 255)

        # Apply Mapping
        for old_id, new_id in MAPPING.items():
            new_mask[mask_arr == old_id] = new_id

        # Save
        new_img = Image.fromarray(new_mask.astype(np.uint8), mode='P')
        new_img.save(dst_path)

if __name__ == '__main__':
    convert_folder(TRAIN_LIST, SRC_TRAIN_DIR, DST_TRAIN_DIR)
    convert_folder(VAL_LIST, SRC_VAL_DIR, DST_VAL_DIR)
    print("\nConversion Complete!")
