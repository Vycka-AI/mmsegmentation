# --- 0. CUSTOM IMPORT ---
custom_imports = dict(imports=['custom_loading'], allow_failed_imports=False)

_base_ = [
    # We removed the deeplabv3plus model base
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

# --- 1. Classes ---
metainfo = dict(
    classes=('background', 'chair', 'human'),
    palette=[
        [0, 0, 0],       # Background
        [255, 0, 0],     # Chair
        [0, 255, 0]      # Human
    ]
)

# --- 2. Model (U-NET) ---
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32), # Critical for U-Net alignment
    
    pretrained=None,
    
    # U-Net Backbone (Contains both Encoder and Decoder paths)
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),

    # Main Decode Head
    decode_head=dict(
        type='FCNHead',
        in_channels=64,   # Matches base_channels of UNet
        in_index=4,       # Index 4 is the final upsampled output of the UNet backbone
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,    # Your 3 classes
        norm_cfg=norm_cfg,
        align_corners=False,
        
        # Kept your OHEM Sampler settings
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0, 
            avg_non_ignore=True)),

    # Auxiliary Head (Attached to an intermediate stage for deep supervision)
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,  # UNet stage 3 output usually has 128 channels here
        in_index=3,       # One stage before the final output
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.4, 
            avg_non_ignore=True)),

    # Train/Test configs
    train_cfg=dict(),
    test_cfg=dict(mode='whole') # 'whole' creates sharper results than 'slide' for U-Net
)

# --- 3. PIPELINES ---
dataset_type = 'BaseSegDataset'
data_root = 'data/eirt_output/batch01'

MY_LABEL_MAP = {0: 0, 2: 1, 6: 2, 1: 255}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsNumpy', label_map=MY_LABEL_MAP),
    dict(type='RandomResize', scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True),
    
    # CHANGED: 769 -> 768 (Better for U-Net division by 32)
    dict(type='RandomCrop', crop_size=(768, 768), cat_max_ratio=0.75),
    
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
        ann_file='train.txt',
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
        ann_file='val.txt',
        img_suffix='.png',
        seg_map_suffix='.npy',
        pipeline=test_pipeline,
        metainfo=metainfo))

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metric='mIoU')
test_evaluator = val_evaluator

# --- 5. OPTIMIZER ---
# Kept exactly as you had it
optim_wrapper = dict(
    _delete_=True,  # <--- THIS IS THE CRITICAL FIX
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999)),
    loss_scale='dynamic')
