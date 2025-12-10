import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mmseg.apis import init_model, inference_model
from mmengine import Config
from tqdm import tqdm
import sys

# --- CONFIGURATION ---
WORK_DIR = 'work_dirs/3_Class_Debug'
CONFIG_FILE = 'configs/3_Class.py'
CHECKPOINT_FILE = 'work_dirs/3_Class_Debug/iter_4000.pth' # Update this to your best .pth

# Data Paths
VAL_LIST = 'data/eirt_output/batch01/val.txt'
IMAGE_DIR = 'data/eirt_output/batch01/rgb'
MASK_DIR = 'data/eirt_output/batch01/mask'
OUTPUT_DIR = 'comparison_results'

# Number of images to process
NUM_SAMPLES = 20

# Class Colors for Visualization (RGB)
# 0: Background (Black)
# 1: Chair (Red)
# 2: Human (Green)
PALETTE = np.array([
    [0, 0, 0],       # 0: Background
    [255, 0, 0],     # 1: Chair
    [0, 255, 0]      # 2: Human
], dtype=np.uint8)

# Your Raw NPY Mapping
# 0->0, 2->1 (Chair), 6->2 (Human)
RAW_ID_MAP = {0: 0, 2: 1, 6: 2} 
# ---------------------

def load_and_map_gt_mask(npy_path):
    """Loads the NPY mask and converts 0,2,6 -> 0,1,2"""
    if not os.path.exists(npy_path):
        return None
        
    raw_mask = np.load(npy_path)
    
    # Fix Shape (H,W,3) -> (H,W)
    if len(raw_mask.shape) == 3:
        raw_mask = raw_mask[:, :, 0] # Take 1st channel
        
    # Remap to Train IDs (0, 1, 2)
    # Default to 255 (Ignore)
    gt_map = np.full_like(raw_mask, 255, dtype=np.uint8)
    
    for raw_id, train_id in RAW_ID_MAP.items():
        gt_map[raw_mask == raw_id] = train_id
        
    return gt_map

def colorize_mask(mask):
    """Converts a 2D Mask (IDs) to 3D RGB Image using the Palette"""
    # Initialize Black
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for label_id, color in enumerate(PALETTE):
        color_mask[mask == label_id] = color
        
    return color_mask

def main():
    # 1. Init Model
    # Need custom imports for the config to load without error
    try:
        import custom_loading 
    except ImportError:
        pass # If fails, we might still be okay if we manually override test pipeline

    cfg = Config.fromfile(CONFIG_FILE)
    # Force simple pipeline for inference to avoid NPY loader issues
    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
        dict(type='PackSegInputs')
    ]
    
    print("Building model...")
    model = init_model(cfg, CHECKPOINT_FILE, device='cuda:0')

    # 2. Prepare Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. Load List
    with open(VAL_LIST, 'r') as f:
        file_ids = [x.strip() for x in f.readlines() if x.strip()]
    
    print(f"Generating comparisons for {min(len(file_ids), NUM_SAMPLES)} images...")

    # 4. Loop
    for file_id in tqdm(file_ids[:NUM_SAMPLES]):
        img_path = os.path.join(IMAGE_DIR, file_id + '.png')
        mask_path = os.path.join(MASK_DIR, file_id + '.npy')
        
        if not os.path.exists(img_path):
            continue

        # A. Run Prediction
        result = inference_model(model, img_path)
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy() # Get 2D array

        # B. Load Ground Truth
        gt_mask = load_and_map_gt_mask(mask_path)
        if gt_mask is None:
            print(f"Missing mask for {file_id}")
            continue

        # C. Visualize
        # Load Original Image
        original_img = np.array(Image.open(img_path))
        
        # Colorize Masks
        vis_pred = colorize_mask(pred_mask)
        vis_gt = colorize_mask(gt_mask)

        # D. Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Col 1: Original
        axes[0].imshow(original_img)
        axes[0].set_title(f"Input: {file_id}")
        axes[0].axis('off')
        
        # Col 2: Ground Truth
        axes[1].imshow(vis_gt)
        axes[1].set_title("Ground Truth\n(Red=Chair, Green=Human)")
        axes[1].axis('off')
        
        # Col 3: Prediction
        axes[2].imshow(vis_pred)
        axes[2].set_title("Prediction")
        axes[2].axis('off')

        # Save
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{file_id}_compare.jpg"))
        plt.close()

    print(f"\nDone! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
