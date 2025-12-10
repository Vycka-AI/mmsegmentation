import os
import glob
import sys
import mmcv
from mmengine import Config
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine.utils import scandir

# --- CRITICAL: Register the custom loader ---
# Even though we are only reading images, the config parser checks for this class.
try:
    import custom_loading
    print("✅ Custom loader registered.")
except ImportError:
    print("❌ Error: Could not import 'custom_loading.py'. Make sure it is in this folder.")
    sys.exit(1)

def find_latest_checkpoint(work_dir):
    """Finds the .pth file with the latest modification time."""
    if not os.path.exists(work_dir):
        print(f"Error: Work dir {work_dir} does not exist.")
        return None
        
    checkpoint_files = list(scandir(work_dir, suffix='.pth'))
    if not checkpoint_files:
        return None

    latest_file = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(work_dir, f)))
    return os.path.join(work_dir, latest_file)

# --- 1. Configuration ---
# Update these paths to match your actual folder structure
WORK_DIR = 'work_dirs/3_Class_Debug'
CONFIG_FILE = 'configs/3_Class.py'

# Input Data
FILTERED_LIST = 'data/real/val.txt' # Your list of IDs
IMAGE_DIR = 'data/real/Real_imgs'         # Your PNG images
NUM_IMAGES_TO_TEST = 10

# --- 2. Find Checkpoint ---
print(f"Searching for checkpoint in {WORK_DIR}...")
latest_checkpoint = find_latest_checkpoint(WORK_DIR)

if latest_checkpoint:
    print(f"Found Checkpoint: {latest_checkpoint}")
else:
    print(f"Error: No .pth checkpoint found in {WORK_DIR}")
    exit()

# --- 3. Build the Model (With Pipeline Fix) ---
print("Building model...")

# Load config object first so we can modify it
cfg = Config.fromfile(CONFIG_FILE)

# FORCE the test pipeline to be simple (Image Only)
# This ensures it doesn't try to load .npy masks that don't exist during inference
cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    dict(type='PackSegInputs')
]

# Initialize with the modified config
model = init_model(cfg, latest_checkpoint, device='cuda:0')
print("Model built successfully.")

# --- 4. Read Image List ---
if not os.path.exists(FILTERED_LIST):
    print(f"Error: List file not found at {FILTERED_LIST}")
    exit()

with open(FILTERED_LIST, 'r') as f:
    file_ids = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(file_ids)} images. Inferring on first {NUM_IMAGES_TO_TEST}...")

# --- 5. Run Inference ---
# Create an output folder for clean results
out_dir = os.path.join(WORK_DIR, 'inference_results')
os.makedirs(out_dir, exist_ok=True)



for file_id in file_ids[:NUM_IMAGES_TO_TEST]:
    # Construct full image path
    img_path = os.path.join(IMAGE_DIR, file_id + '.jpg')

    if not os.path.exists(img_path):
        print(f"Skipping missing image: {img_path}")
        continue

    print(f"Processing: {file_id}...")

    # Run Inference
    result = inference_model(model, img_path)

    # Save Result
    out_file = os.path.join(out_dir, f"{file_id}_pred.jpg")
    
    # Visualization
    show_result_pyplot(
        model,
        img_path,
        result,
        show=False,
        out_file=out_file,
        opacity=0.5 
    )

print(f"\nDone! Results saved to: {out_dir}")
