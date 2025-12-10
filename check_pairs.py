import cv2
import numpy as np
import os

# --- Configuration ---
# These paths point to the data lists and folders in your project
list_file  = 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/aug.txt'
img_dir    = 'data/VOCdevkit/VOC2012/JPEGImages'
mask_dir   = 'data/VOCdevkit/VOC2012/SegmentationClassAug'

# --- PASCAL VOC Color Palette (for nice visualization) ---
# 0=background, 1=aeroplane, 2=bicycle, ..., 20=tvmonitor
palette = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
    [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 192, 0], [128, 192, 0], [0, 64, 128]
]

# Create a simple numpy array for the color palette
color_palette = np.array(palette, dtype=np.uint8)


def colorize_mask(mask):
    """Converts a grayscale label mask to a color image."""
    # Create an empty 3-channel (color) image
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Apply colors for each class ID
    for class_id in range(1, len(color_palette)): # Skip background
        color_mask[mask == class_id] = color_palette[class_id]

    # Highlight the 'ignore' label (255) in white
    color_mask[mask == 255] = [255, 255, 255]
    return color_mask


# --- Main Script ---
try:
    with open(list_file, 'r') as f:
        # Read all filenames (e.g., "2008_002768") from the aug.txt
        file_ids = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"Error: Could not find the file list: {list_file}")
    print("Please make sure you ran the 'ls ... > aug.txt' command correctly.")
    exit()

if not file_ids:
    print(f"Error: The file {list_file} is empty!")
    exit()

print(f"Loaded {len(file_ids)} file IDs from {list_file}.")
print("Press 'q' or 'Esc' to quit.")
print("Press any other key to view the next pair.")

index = 0
while True:
    file_id = file_ids[index]

    # 1. Create the full paths
    img_path = os.path.join(img_dir, file_id + '.jpg')
    mask_path = os.path.join(mask_dir, file_id + '.png')

    # 2. Load both files
    image = cv2.imread(img_path)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 3. Check if files were loaded
    if image is None:
        print(f"Warning: Could not load image: {img_path}")
        index = (index + 1) % len(file_ids) # Skip to next
        continue

    if mask_gray is None:
        print(f"Warning: Could not load mask: {mask_path}")
        index = (index + 1) % len(file_ids) # Skip to next
        continue

    # 4. Create the visualization
    mask_color = colorize_mask(mask_gray)

    # Resize mask to match image (using INTER_NEAREST to keep labels sharp)
    h, w, _ = image.shape
    mask_color = cv2.resize(mask_color, (w, h), interpolation=cv2.INTER_NEAREST)

    # Stack them side-by-side
    combined_view = np.hstack([image, mask_color])

    # Add text
    cv2.putText(combined_view, f"Image: {file_id}.jpg", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_view, f"Mask: {file_id}.png (Classes: {np.unique(mask_gray)})", (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 5. Show the image
    cv2.imshow("Image (Left)  |  Mask (Right)", combined_view)

    # 6. Wait for key press
    key = cv2.waitKey(0)

    if key == ord('q') or key == 27:  # 'q' or 'Esc'
        break
    else:
        # Go to the next image
        index = (index + 1) % len(file_ids)

cv2.destroyAllWindows()
