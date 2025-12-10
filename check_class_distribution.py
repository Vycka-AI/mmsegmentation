import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
CLASSES = {
    0: "Background",
    1: "Person",
    2: "Chair",
    3: "Diningtable"
}

# Define the datasets to check
DATA_SPECS = [
    {
        "name": "TRAINING SET",
        "list_path": "data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
        "mask_dir": "data/VOCdevkit/VOC2012/SegmentationClass4Class_Train"
    },
    {
        "name": "VALIDATION SET",
        "list_path": "data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
        "mask_dir": "data/VOCdevkit/VOC2012/SegmentationClass4Class_Val"
    }
]

def analyze_dataset(dataset_info):
    print(f"\n--- Analyzing {dataset_info['name']} ---")
    
    list_path = dataset_info["list_path"]
    mask_dir = dataset_info["mask_dir"]

    if not os.path.exists(list_path):
        print(f"Error: File list not found: {list_path}")
        return

    with open(list_path, 'r') as f:
        files = [x.strip() for x in f.readlines()]

    print(f"Scanning {len(files)} images from {list_path}...")

    if len(files) == 0:
        print("Error: The file list is empty!")
        return

    # Initialize counters
    class_stats = {k: {'images': 0, 'pixels': 0} for k in CLASSES.keys()}
    class_stats[255] = {'images': 0, 'pixels': 0}

    for file_name in tqdm(files):
        mask_path = os.path.join(mask_dir, file_name + ".png")
        
        try:
            # Open using PIL to get raw index values
            mask = Image.open(mask_path)
            mask_arr = np.array(mask)
        except FileNotFoundError:
            # This is expected for some files in standard VOC lists 
            continue

        # Get unique values and counts
        unique, counts = np.unique(mask_arr, return_counts=True)
        
        for val, count in zip(unique, counts):
            if val in class_stats:
                class_stats[val]['pixels'] += count
                class_stats[val]['images'] += 1
            else:
                pass

    # Print Results
    print(f"\nResults for {dataset_info['name']}:")
    print(f"{'ID':<5} {'Class Name':<15} {'Images (Count)':<15} {'Images (%)':<15} {'Total Pixels':<15}")
    print("-" * 70)
    
    total_imgs = len(files)
    
    for cls_id in sorted(CLASSES.keys()):
        name = CLASSES[cls_id]
        imgs = class_stats[cls_id]['images']
        pct = (imgs / total_imgs) * 100 if total_imgs > 0 else 0
        pixels = class_stats[cls_id]['pixels']
        print(f"{cls_id:<5} {name:<15} {imgs:<15} {pct:<15.2f}% {pixels:<15}")

    print("-" * 70)
    # Avoid division by zero if total_imgs is somehow 0
    ignore_pct = (class_stats[255]['images']/total_imgs)*100 if total_imgs > 0 else 0
    print(f"255   Ignore/Void     {class_stats[255]['images']:<15} {ignore_pct:<15.2f}% {class_stats[255]['pixels']:<15}")

if __name__ == "__main__":
    # This was the typo: changed DATASETS to DATA_SPECS
    for ds in DATA_SPECS:
        analyze_dataset(ds)
