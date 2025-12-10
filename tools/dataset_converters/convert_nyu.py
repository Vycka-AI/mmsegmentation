# Save this file as:
# ~/mm_segment/mmsegmentation/tools/dataset_converters/convert_nyu.py

import h5py
import numpy as np
import os
import argparse
from PIL import Image
import os.path as osp
from mmengine.utils import mkdir_or_exist

def main():
    parser = argparse.ArgumentParser(description='Convert NYUv2 dataset from .mat file')
    parser.add_argument('mat_file', help='path to the nyu_depth_v2_labeled.mat file')
    parser.add_argument('-o', '--out_dir', help='output path', default='data/nyu')
    args = parser.parse_args()

    mat_path = args.mat_file
    out_dir = args.out_dir

    print(f"Loading {mat_path}...")
    try:
        f = h5py.File(mat_path, 'r')
    except Exception as e:
        print(f"Error opening .mat file: {e}")
        print("Please ensure h5py is installed (`pip install h5py`) and the file is not corrupt.")
        return

    print("Making directories...")
    mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    # According to the official split, 795 for training and 654 for validation
    train_split = 795

    print("Generating images and annotations...")
    total_images = len(f['images'])
    
    for i in range(total_images):
        img = f['images'][i]
        label = f['labels'][i]

        # h5py reads in (channels, height, width)
        # We need (height, width, channels) for PIL
        img = np.transpose(img, (2, 1, 0))
        # Labels are (width, height)
        # We need (height, width)
        label = np.transpose(label, (1, 0))

        img = Image.fromarray(img.astype('uint8'), 'RGB')
        # Labels are 0-indexed, which is what mmseg wants (0 is 'unlabeled')
        label = Image.fromarray(label.astype('uint8'), 'P')

        if (i + 1) <= train_split:
            phase = 'training'
        else:
            phase = 'validation'

        filename = f"{i:05d}.png"
        img.save(osp.join(out_dir, 'images', phase, filename))
        label.save(osp.join(out_dir, 'annotations', phase, filename))

        if (i + 1) % 100 == 0 or (i + 1) == total_images:
            print(f"Processed {i+1}/{total_images} images...")

    f.close()
    print("Done!")
    print(f"Data successfully generated in {out_dir}")

if __name__ == '__main__':
    main()
