_base_ = './schedule_40k.py'

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper', loss_scale='dynamic',
    # # https://massedcompute.com/faq-answers/?question=What%20are%20the%20supported%20NVIDIA%20GPUs%20for%20BF16
    # dtype='bfloat16',  # A100 support bf16. V100 not support, in this case pls use `float16`
    # type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05),
    # constructor='mmpretrain.LearningRateDecayOptimWrapperConstructorV2',
    paramwise_cfg=dict(
        # layer_decay_rate=0.65,  # for LearningRateDecayOptimWrapperConstructor use
        custom_keys={  ## custom_keys for ViTs
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
        }),
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

auto_scale_lr = dict(base_batch_size=2, enable=True)
randomness = dict(seed=0, diff_rank_seed=True)