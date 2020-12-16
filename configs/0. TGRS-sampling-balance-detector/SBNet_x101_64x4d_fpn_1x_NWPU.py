# model settings
#--out results.pkl
#loss_cls=dict( type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.005)
model = dict(
    type='CascadeRCNN',
    num_stages=3,
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            #loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            loss_bbox=dict(type='BalancedL1Loss', alpha=0.5, gamma=1.5, beta=1.0,loss_weight=1.0)
        ),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            #loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            loss_bbox=dict(type='BalancedL1Loss', alpha=0.5, gamma=1.5, beta=1.0,loss_weight=1.0)
        ),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='BalancedL1Loss', alpha=0.5, gamma=1.5, beta=1.0,loss_weight=1.0)
            #loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        )
    ])
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            #sampler=dict(type='RandomSampler',num=512,pos_fraction=0.25,neg_pos_ub=-1,add_gt_as_proposals=True),
            sampler=dict(
                        type='CombinedSampler',
                        num=512,
                        pos_fraction=0.25,
                        add_gt_as_proposals=True,
                        pos_sampler=dict(type='InstanceBalancedPosSampler'),
                        neg_sampler=dict(
                            type='IoUBalancedNegSampler',
                            floor_thr=-1,
                            floor_fraction=0,
                            num_bins=3)),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            #sampler=dict(type='RandomSampler',num=512,pos_fraction=0.25,neg_pos_ub=-1,add_gt_as_proposals=True),
            sampler=dict(
                        type='CombinedSampler',
                        num=512,
                        pos_fraction=0.25,
                        add_gt_as_proposals=True,
                        pos_sampler=dict(type='InstanceBalancedPosSampler'),
                        neg_sampler=dict(
                            type='IoUBalancedNegSampler',
                            floor_thr=-1,
                            floor_fraction=0,
                            num_bins=3)),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            # sampler=dict(type='RandomSampler',num=512,pos_fraction=0.25,neg_pos_ub=-1,add_gt_as_proposals=True),
            sampler=dict(
                type='CombinedSampler',
                num=512,
                pos_fraction=0.25,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
# dataset settings
#NWPU
dataset_type = 'VOCDataset'
data_root = '/home/han/Desktop/DOTA/mmdetection-master/data/VOCdevkit/VOC2007/'
data_root_test = '/home/han/Desktop/DOTA/mmdetection-master/data/VOCdevkit/VOC2007/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #multi-scale training
    dict(
        type='Resize',
        #img_scale=(1400, 1100),
        img_scale=[(1400, 1100),(1333, 800), (1000, 800),(800, 600),(600,400)],
        multiscale_mode='value',
        keep_ratio=True),
    #single-scale training
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        #img_scale=[(1400, 1100),(1333, 800), (1000, 800),(800, 600),(600,400)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Mains/trainval.txt',
        img_prefix=data_root ,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Mains/trainval.txt',
        img_prefix=data_root ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Mains/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[40, 46])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 48
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_rcnn_x101_64x4d_fpn_1x'
load_from = None
resume_from =None
workflow = [('train', 1)]
