import os
import glob
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine.utils import scandir

def find_latest_checkpoint(work_dir):
    """Finds the .pth file with the latest modification time in a dir."""
    # Get all .pth file paths
    checkpoint_files = list(scandir(work_dir, suffix='.pth'))
    
    if not checkpoint_files:
        return None

    # Find the latest one based on modification time
    latest_file = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(work_dir, f)))
    return os.path.join(work_dir, latest_file)

# --- 1. Configuration ---
WORK_DIR = 'work_dirs/3_Class_Debug'
CONFIG_FILE = 'configs/3_Class.py'
#ImageSets/Segmentation/train_3class.txt
FILTERED_LIST = 'data/eirt_output/batch01/val.txt'
IMAGE_DIR = 'data/eirt_output/batch01/rgb'
NUM_IMAGES_TO_TEST = 20

# --- 2. Find Latest Checkpoint ---
print(f"Searching for latest checkpoint in {WORK_DIR}...")
latest_checkpoint = find_latest_checkpoint(WORK_DIR)

if latest_checkpoint:
    print(f"Found: {latest_checkpoint}")
else:
    print(f"Error: No checkpoints (.pth files) found in {WORK_DIR}")
    exit()

# --- 3. Build the Model ---
print("Building model...")
# (Use 'cpu' if you don't have a GPU)
model = init_model(CONFIG_FILE, latest_checkpoint, device='cuda:0')
print("Model built successfully.")

# --- 4. Read Filtered Image List ---
try:
    with open(FILTERED_LIST, 'r') as f:
        # Read all filenames (e.g., "2007_000032") from the list
        file_ids = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"Error: Cannot find filtered list file: {FILTERED_LIST}")
    exit()

if not file_ids:
    print(f"Error: The file list {FILTERED_LIST} is empty.")
    exit()

print(f"Loaded {len(file_ids)} images from the filtered list. Running inference on the first {NUM_IMAGES_TO_TEST}...")

# --- 5. Run Inference on Images ---
for file_id in file_ids[:NUM_IMAGES_TO_TEST]:
    img_path = os.path.join(IMAGE_DIR, file_id + '.png')
    
    if not os.path.exists(img_path):
        print(f"Warning: Image not found, skipping: {img_path}")
        continue
    
    print(f"\nProcessing: {img_path}")
    
    # Run the model
    result = inference_model(model, img_path)
    
    # Define an output file name
    out_file = os.path.join(WORK_DIR, f"{file_id}_3class_result.jpg")
    
    # Save the visualization (without popping up a window)
    show_result_pyplot(
        model,
        img_path,
        result,
        show=False, 
        out_file=out_file,
        opacity=0.6 # Set opacity for the segmentation mask
    )
    print(f"Saved result to: {out_file}")

print("\nInference complete.")
