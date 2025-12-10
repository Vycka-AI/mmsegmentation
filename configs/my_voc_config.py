# Save this file as:
# ~/mm_segment/mmsegmentation/configs/my_voc_config.py

# --- 1. Inherit all the basics ---
_base_ = [
    './_base_/models/pspnet_r50-d8.py',
    './_base_/datasets/pascal_voc12_aug.py', # Use the AUGMENTED voc config
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'    # 20k iterations is enough for VOC
]

# --- 2. Modify the Model (THIS IS THE FIX) ---
model = dict(
    # This block defines the preprocessor
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32  # <-- This line fixes the error
    ),
    pretrained='open-mmlab://resnet50_v1c',
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21)
)

# --- 3. Modify the Dataloader ---
train_dataloader = dict(
    batch_size=4,  # Lower to 2 if you run out of memory
    num_workers=4
)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=100)
)
