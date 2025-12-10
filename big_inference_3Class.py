import os
import glob
import sys
import mmcv
from PIL import Image  # <--- ADDED for resolution checking
from mmengine import Config
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine.utils import scandir

# --- CRITICAL: Register the custom loader ---
try:
    import custom_loading
    print("✅ Custom loader registered.")
except ImportError:
    print("❌ Error: Could not import 'custom_loading.py'. Make sure it is in this folder.")
    sys.exit(1)

def find_latest_checkpoint(work_dir):
    if not os.path.exists(work_dir):
        print(f"Error: Work dir {work_dir} does not exist.")
        return None

    checkpoint_files = list(scandir(work_dir, suffix='.pth'))
    if not checkpoint_files:
        return None

    latest_file = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(work_dir, f)))
    return os.path.join(work_dir, latest_file)

# --- 1. Configuration ---
WORK_DIR = 'work_dirs/3_Class_Debug'
CONFIG_FILE = 'configs/3_Class.py'

# Input Data
FILTERED_LIST = 'data/real/val.txt'
IMAGE_DIR = 'data/real/Real_imgs'
NUM_IMAGES_TO_TEST = 10

# --- 2. Find Checkpoint ---
print(f"Searching for checkpoint in {WORK_DIR}...")
latest_checkpoint = find_latest_checkpoint(WORK_DIR)

if latest_checkpoint:
    print(f"Found Checkpoint: {latest_checkpoint}")
else:
    print(f"Error: No .pth checkpoint found in {WORK_DIR}")
    exit()

# --- 3. Build the Model ---
print("Building model...")
cfg = Config.fromfile(CONFIG_FILE)

# Force simple pipeline (Image Only)
cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    dict(type='PackSegInputs')
]

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
out_dir = os.path.join(WORK_DIR, 'inference_results')
os.makedirs(out_dir, exist_ok=True)

# Temp folder for resized images
temp_resize_dir = os.path.join(WORK_DIR, 'temp_resized_inputs')
os.makedirs(temp_resize_dir, exist_ok=True)

for file_id in file_ids[:NUM_IMAGES_TO_TEST]:
    # Try JPG first, then PNG if needed (or rely on list to perform check)
    # Assuming JPG based on your snippet, but let's be robust
    img_path = os.path.join(IMAGE_DIR, file_id + '.jpg')
    if not os.path.exists(img_path):
        img_path = os.path.join(IMAGE_DIR, file_id + '.png')
    
    if not os.path.exists(img_path):
        print(f"Skipping missing image: {img_path}")
        continue

    print(f"Processing: {file_id}...", end=" ")

    # --- NEW: Check Size and Resize if needed ---
    target_path = img_path
    
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            # Check if larger than Full HD (1920x1080)
            if w > 1920 or h > 1080:
                print(f"\n   ⚠️ Too large ({w}x{h}). Resizing to fit 1920x1080...", end="")
                
                # 'thumbnail' resizes in place, preserving aspect ratio
                # It ensures the result fits WITHIN the given size
                img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
                
                # Save to temp location so MMSeg can load it from file
                temp_path = os.path.join(temp_resize_dir, f"{file_id}_resized.jpg")
                img.save(temp_path)
                
                # Update target path to point to the resized image
                target_path = temp_path
    except Exception as e:
        print(f"Error reading image size: {e}")
        continue
    # -------------------------------------------

    # Run Inference (On the potentially resized file)
    result = inference_model(model, target_path)

    # Save Result
    out_file = os.path.join(out_dir, f"{file_id}_pred.jpg")

    show_result_pyplot(
        model,
        target_path, # Use the image we actually inferred on for visualization overlay
        result,
        show=False,
        out_file=out_file,
        opacity=0.5
    )
    print(" Done.")

print(f"\nDone! Results saved to: {out_dir}")
