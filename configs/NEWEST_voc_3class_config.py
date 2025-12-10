# Save this file as:
# ~/mm_segment/mmsegmentation/configs/my_voc_3class_config.py

# --- 1. Inherit basics (model, runtime, schedule) ---
_base_ = [
    './_base_/models/pspnet_r50-d8.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

# --- 2. Define Our 3 Classes ---
# This is the only information the dataset needs to create the label map.
metainfo = dict(
    classes=('person', 'chair', 'diningtable'),  # <-- Use correct name 'diningtable'
    palette=[[220, 20, 60], [119, 11, 32], [0, 0, 142]]
)

# --- 3. Set Model Heads to 3 Classes ---
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
    decode_head=dict(num_classes=3),     # <-- Set to 3
    auxiliary_head=dict(num_classes=3) # <-- Set to 3
)

# --- 4. Define Pipelines (No 'label_map' needed) ---
dataset_type = 'PascalVOCDataset'  # <-- Use the correct dataset type
data_root = 'data/VOCdevkit/VOC2012'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),  # <-- No 'label_map' here
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),  # <-- No 'label_map' here
    dict(type='PackSegInputs')
]

# --- 5. Define Dataloaders (No 'label_map' needed) ---
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='SegmentationClassAug'),
        ann_file='ImageSets/Segmentation/train_3class.txt',
        pipeline=train_pipeline,
        metainfo=metainfo,  # <-- Just pass metainfo
	#filter_cfg=dict(min_size=1)
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
            seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val_3class.txt',
        pipeline=test_pipeline,
        metainfo=metainfo  # <-- Just pass metainfo
    ))

test_dataloader = val_dataloader

# --- 6. Define Evaluator ---
val_evaluator = dict(type='IoUMetric', iou_metric='mIoU')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50)
)

# --- Add this block to the end of your config file ---

# Enable TensorBoard logging
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Change 'OptimWrapper' to 'AmpOptimWrapper'
optim_wrapper = dict(
    type='AmpOptimWrapper',  # <--- This enables mixed precision
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    loss_scale='dynamic'
)
