_base_ = [
    '../_base_/models/dinov3_vit-large_segmenter.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/adamw_40k.py'
]

work_dir = "work_dirs/segmentation/potsdam/{{fileBasenameNoExtension}}_3"

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MsReDINOv3', freeze=True,
        msre_config=dict(
            token_length=100, embed_dims=1024,
            num_layers=24, patch_size=16,
        ),
    ),
    decode_head=dict(num_classes={{_base_.num_classes}}),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),

    paramwise_cfg=dict(
        custom_keys={  ## custom_keys for ViTs
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            'learnable_tokens':dict(decay_mult=0.0),
            'level_encoding':dict(decay_mult=0.0),
        }),
)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)

default_hooks = dict(
    checkpoint = dict(max_keep_ckpts=1, save_best=['mIoU'],interval=2000),
)