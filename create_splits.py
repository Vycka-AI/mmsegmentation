import os
import random
import glob

# --- CONFIG ---
# Path to your images (PNGs)
data_root = './'
img_dir = 'rgb'  # The folder name inside data_root
# --------------

def generate_splits():
    full_img_path = os.path.join(data_root, img_dir)
    
    # Get all filenames without extension
    # We scan for .png (or .jpg)
    files = glob.glob(os.path.join(full_img_path, "*.png"))
    
    if not files:
        print(f"Error: No images found in {full_img_path}")
        return

    print(f"Found {len(files)} images.")
    
    # Extract just the ID (filename without extension)
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    
    # Shuffle
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(file_ids)
    
    # Calculate Split
    total = len(file_ids)
    val_size = int(total * 0.10) # 10% for Eval
    
    val_ids = file_ids[:val_size]
    train_ids = file_ids[val_size:]
    
    # Save
    with open(os.path.join(data_root, 'val.txt'), 'w') as f:
        f.write('\n'.join(sorted(val_ids)))
        
    with open(os.path.join(data_root, 'train.txt'), 'w') as f:
        f.write('\n'.join(sorted(train_ids)))

    print(f"\n--- SPLIT COMPLETE ---")
    print(f"Training:   {len(train_ids)} images (90%)")
    print(f"Validation: {len(val_ids)} images (10%)")
    print(f"Lists saved to {data_root}")

if __name__ == "__main__":
    generate_splits()
