import cv2
import numpy as np
import glob
import os

folder_path = 'data/nyu/annotations/training/*.png'
mask_files = sorted(glob.glob(folder_path))

if not mask_files:
    print(f"Error: No annotation files found at {folder_path}")
    exit()

print("Press 'q' or 'Esc' to quit.")
print("Press any other key to view the next mask.")

index = 0
while True:
    file_path = mask_files[index]
    
    # Read the image in its original, single-channel format
    mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) # Use UNCHANGED

    if mask is None:
        print(f"Warning: Could not read image from {file_path}. Skipping.")
    else:
        # --- THIS IS THE TEST ---
        # Find all unique pixel values in the mask
        unique_values = np.unique(mask)
        print(f"File: {os.path.basename(file_path)} | Unique values: {unique_values}")
        # ------------------------
        
        # Scale values to 0-255 so the colormap works
        scaled_mask = (mask * 18).astype(np.uint8)
        color_mask = cv2.applyColorMap(scaled_mask, cv2.COLORMAP_JET)

        # Add the filename text to the image
        file_name = os.path.basename(file_path)
        cv2.putText(color_mask, file_name, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(color_mask, f"Values: {unique_values}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Colorized Annotation Mask", color_mask)

    key = cv2.waitKey(0)

    if key == ord('q') or key == 27:  # 'q' or 'Esc' key
        break
    else:
        index = (index + 1) % len(mask_files)

cv2.destroyAllWindows()
