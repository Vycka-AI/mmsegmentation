# Save/Overwrite as custom_loading.py
import numpy as np
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class LoadAnnotationsNumpy(BaseTransform):
    def __init__(self, 
                 # Default mapping based on your inspection
                 # You can swap 1 and 2 if '2' turns out to be Chair instead of Human
                 label_map={0: 0, 2: 1, 6: 2}, 
                 reduce_zero_label=False):
        self.label_map = label_map
        self.reduce_zero_label = reduce_zero_label

    def transform(self, results):
        seg_path = results['seg_map_path']
        
        # Load NPY
        mask = np.load(seg_path)
        
        # --- 1. FIX SHAPE ---
        # Input shape is (1080, 1920, 3). We need (1080, 1920).
        if len(mask.shape) == 3:
            if mask.shape[2] == 3:  # It's RGB, take Channel 0
                mask = mask[:, :, 0]
            elif mask.shape[2] == 1: # It's (H, W, 1)
                mask = mask.squeeze(2)
            elif mask.shape[0] == 1: # It's (1, H, W)
                mask = mask.squeeze(0)
        
        # --- 2. REMAP VALUES ---
        # Create blank mask
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        
        for old_val, new_id in self.label_map.items():
            # Exact match since these are integers [0, 2, 6]
            new_mask[mask == old_val] = new_id

        results['gt_seg_map'] = new_mask
        results['seg_fields'].append('gt_seg_map')
        
        return results
