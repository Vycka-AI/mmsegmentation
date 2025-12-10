# Save this file as: ~/mm_segment/mmsegmentation/configs/my_nyu_config.py

# --- 1. Inherit the basics ---
_base_ = [
    './_base_/models/pspnet_r50-d8.py',     # The PSPNet model
    './_base_/default_runtime.py',         # Logging, checkpoints, etc.
    './_base_/schedules/schedule_80k.py'  # 80k iterations
]

# --- 2. Define Dataset Info ---
data_root = 'data/nyu'
dataset_type = 'BaseSegDataset' # Use the generic dataset class
crop_size = (480, 480)

# --- 3. THE FIX: Define Dataset Metainfo (Class Names) ---
# We provide the 13 class names for NYUv2
metainfo = dict(
    classes=('bed', 'book', 'cabinet', 'chair', 'floor', 'furniture', 'object',
             'picture', 'sofa', 'table', 'tv', 'wall', 'window'),
    palette=[[174, 199, 232], [152, 223, 138], [31, 119, 180],
             [255, 187, 120], [188, 189, 34], [140, 86, 75], [255, 152, 150],
             [214, 39, 40], [197, 176, 213], [148, 103, 189], [196, 156, 148],
             [23, 190, 207], [247, 182, 210]])

# --- 4. Modify Model Config ---
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32),
    pretrained='open-mmlab://resnet50_v1c',
    decode_head=dict(num_classes=13),
    auxiliary_head=dict(num_classes=13))

# --- 5. Define Pipelines ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(640, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

# --- 6. Define Dataloaders (Apply the Metainfo) ---
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
        metainfo=metainfo,
	reduce_zero_label=True))  # <-- FIX IS HERE

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        metainfo=metainfo,
	reduce_zero_label=True))  # <-- AND HERE
test_dataloader = val_dataloader

# --- 7. Define Evaluator ---
val_evaluator = dict(type='IoUMetric', iou_metric='mIoU')
test_evaluator = val_evaluator
