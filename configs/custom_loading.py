import numpy as np
import sys
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class LoadAnnotationsNumpy(BaseTransform):
    def __init__(self, 
                 label_map=None, 
                 reduce_zero_label=False):
        # Default map if not provided in config
        if label_map is None:
            self.label_map = {0: 0, 2: 1, 6: 2, 1: 255}
        else:
            self.label_map = label_map
            
        self.reduce_zero_label = reduce_zero_label

    def transform(self, results):
        seg_path = results['seg_map_path']
        mask = np.load(seg_path)
        
        # 1. Fix Shape
        if len(mask.shape) == 3:
            if mask.shape[2] == 3: mask = mask[:, :, 0]
            elif mask.shape[2] == 1: mask = mask.squeeze(2)
            elif mask.shape[0] == 1: mask = mask.squeeze(0)
        
        # 2. Defensive Remapping
        # Initialize everything to 255 (Ignore)
        new_mask = np.full_like(mask, 255, dtype=np.uint8)
        
        for old_val, new_id in self.label_map.items():
            new_mask[mask == old_val] = new_id

        # --- DEBUG TRAP START ---
        # Check what values are actually in the new mask
        unique_vals = np.unique(new_mask)
        
        # Allowed: 0, 1, 2 OR 255.
        # Anything else is a crash.
        allowed = {0, 1, 2, 255}
        
        for val in unique_vals:
            if val not in allowed:
                print(f"\n[CRITICAL ERROR DETECTED] inside {seg_path}")
                print(f"Original Values in file: {np.unique(mask)}")
                print(f"Your Map: {self.label_map}")
                print(f"Resulting Bad Value: {val}")
                print("This value will crash the GPU!")
                sys.exit(1) # Stop immediately
        # --- DEBUG TRAP END ---

        results['gt_seg_map'] = new_mask
        results['seg_fields'].append('gt_seg_map')
        
        return results
