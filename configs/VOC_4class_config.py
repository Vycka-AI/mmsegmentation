# Save this as configs/VOC_4class_config.py

# --- 1. Inherit basics ---
_base_ = [
    './_base_/models/pspnet_r50-d8.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

# --- 2. Define Our 4 Classes ---
metainfo = dict(
    classes=('background', 'person', 'chair', 'diningtable'),
    palette=[[0, 0, 0], [220, 20, 60], [119, 11, 32], [0, 0, 142]]
)

# --- 3. Model Config ---
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32),
    pretrained='open-mmlab://resnet18_v1c',
    decode_head=dict(num_classes=4),    # 4 Classes
    auxiliary_head=dict(num_classes=4)  # 4 Classes
)

# --- 4. Pipelines ---
# We use 'BaseSegDataset' to avoid PASCAL specific checks
dataset_type = 'BaseSegDataset'
data_root = 'data/VOCdevkit/VOC2012'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'), 
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# --- 5. Dataloaders (Pointing to NEW folders) ---
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='SegmentationClass4Class_Train'), # <--- NEW FOLDER
        ann_file='ImageSets/Segmentation/train_3class.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
        metainfo=metainfo,
        reduce_zero_label=False 
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='SegmentationClass4Class_Val'), # <--- NEW FOLDER
        ann_file='ImageSets/Segmentation/val_3class.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        metainfo=metainfo,
        reduce_zero_label=False 
    ))

test_dataloader = val_dataloader

# --- 6. Evaluator ---
val_evaluator = dict(type='IoUMetric', iou_metric='mIoU')
test_evaluator = val_evaluator

# --- 7. Optimization (AMP Enabled) ---
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    loss_scale='dynamic'
)

# --- 8. Hooks ---
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=100)
)
train_cfg = dict(val_interval=500)


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')  # <--- This enables TensorBoard
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
