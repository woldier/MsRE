data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=(512, 512),
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DINOv3ViT',image_size=512,
        hidden_size=1024, intermediate_size=4096,
        num_attention_heads=16, num_hidden_layers=24,
        num_register_tokens=4,
        # query_bias= False, key_bias= False, value_bias= False,
        # init_cfg=[
        #     dict(type='TruncNormal', mean=0., std =0.02, bias=0., layer=['Linear', 'Conv2d'],_scope_='mmengine'),
        #     dict(type='Constant',val=1.0, bias=0., layer=['LayerNorm']),
        # ],
        # init_cfg=dict(type='Pretrained', checkpoint='/home/woldier/data/checkpoints/dinov3-vitl16-pretrain-sat493m-hf.pth')
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/dinov3-vitl16-pretrain-lvd1689m-hf.pth')

    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels=1024,
        # TODO Files inheriting from this config must reset num_classes according to the task.
        num_classes=None,
        num_layers=2,
        num_heads=16,
        embed_dims=1024,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)