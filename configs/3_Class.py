# --- 0. CUSTOM IMPORT ---
# Ensure 'custom_loading.py' is in the same folder where you run the command
custom_imports = dict(imports=['custom_loading'], allow_failed_imports=False)

_base_ = [
    './_base_/models/deeplabv3plus_r50-d8.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

# --- 1. Classes ---
# Order matches IDs: 0=Back, 1=Chair, 2=Human
metainfo = dict(
    classes=('background', 'chair', 'human'),
    palette=[
        [0, 0, 0],       # Background (Black)
        [255, 0, 0],     # Chair (Red)
        [0, 255, 0]      # Human (Green)
    ]
)

# --- 2. Model (FIXED) ---
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32),
    decode_head=dict(
        num_classes=3, 
        ignore_index=255,
        
        # FIX: Use OHEM instead of class_weight to handle imbalance
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            # class_weight=[1.0, 10.0, 5.0], <--- DELETED (This was causing the crash)
            avg_non_ignore=True)),
    auxiliary_head=dict(
        num_classes=3,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.4,
            # class_weight=[1.0, 10.0, 5.0], <--- DELETED
            avg_non_ignore=True))
)

# --- 3. PIPELINES ---
dataset_type = 'BaseSegDataset'
data_root = 'data/eirt_output/batch01'

# MAPPING: Original_NPY_Value -> New_Class_ID
# 0 -> 0 (Background)
# 2 -> 1 (Chair)
# 6 -> 2 (Human)
MY_LABEL_MAP = {0: 0, 2: 1, 6: 2, 1: 255}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsNumpy', label_map=MY_LABEL_MAP),
    # Resize slightly to add scale variation
    dict(type='RandomResize', scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True),
    # Crop 769x769 to fit Chair legs in feature map
    dict(type='RandomCrop', crop_size=(769, 769), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    dict(type='LoadAnnotationsNumpy', label_map=MY_LABEL_MAP),
    dict(type='PackSegInputs')
]

# --- 4. DATALOADERS ---
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb',     
            seg_map_path='mask'), 
        ann_file='train.txt',      # Created by split script
        img_suffix='.png', 
        seg_map_suffix='.npy', 
        pipeline=train_pipeline,
        metainfo=metainfo))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb', 
            seg_map_path='mask'),
        ann_file='val.txt',        # Created by split script
        img_suffix='.png',
        seg_map_suffix='.npy',
        pipeline=test_pipeline,
        metainfo=metainfo))

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metric='mIoU')
test_evaluator = val_evaluator

# --- 5. OPTIMIZER ---
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999)),
    loss_scale='dynamic')
