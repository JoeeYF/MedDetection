
from meddet.task import *
from meddet.data.pipelines import *

DIM = 3
TASK = 'DET'
CLASSES = 1
FLOAT16 = True
model = dict(
    type='FasterRCNN',
    dim=DIM,
    backbone=DeepLungBK(
        dim=DIM,
        dcn=dict(type='DCNv2', deformable_groups=1),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=None,
    rpn_head=dict(
        type='RPNHead',
        dim=DIM,
        in_channels=128,
        feat_channels=64,
        level_first=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            base_scales=1,
            scales=[1.25, 2.5, 5.0],
            ratios=[1],
            strides=[4]
        ),
        bbox_coder=dict(
            type='DeltaBBoxCoder',
            target_stds=[1., 1., 1., 1., 1., 1.]),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.8 ** DIM,
            neg_iou_thr=0.2 ** DIM,
            min_pos_iou=0.5 ** DIM,
            match_low_quality=True,
            num_neg=800),
        sampler=dict(
            type='OHEMSampler',
            num=3 * 16,
            pos_fraction=0.35),
        proposal=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=500,
            max_num=200,
            nms_thr=0.7 ** DIM,
            min_bbox_size=0),
        losses=dict(
            cls=CrossEntropyLoss(use_sigmoid=True),
            reg=SmoothL1Loss(beta=1.0, reduction='mean', loss_weight=1.0)),
        metrics=[
            IOU(aggregate='none'),
            Dist(aggregate='none', dist_threshold=5)
        ],
    ),
    roi_head=dict(
        type='ROIHead',
        dim=DIM,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            dim=DIM,
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=128,
            featmap_strides=[4]),
        bbox_head=dict(
            type='ConvFCBBoxHead',
            dim=DIM,
            num_classes=CLASSES,
            in_channels=128,
            feat_channels=128,
            bbox_coder=dict(
                type='DeltaBBoxCoder',
                target_stds=[1., 1., 1., 1., 1., 1.]),
            assigner=dict(
                type='IoUAssigner',
                pos_iou_thr=0.8 ** DIM,
                neg_iou_thr=0.3 ** DIM,
                min_pos_iou=0.4 ** DIM),
            sampler=dict(
                type='OHEMSampler',
                num=16,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_random_gt=1),
            nms=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.15,
                nms_fun=dict(type='nms', iou_threshold=0.1),
                max_per_img=100),
            losses=dict(
                cls=dict(type='CrossEntropyLoss'),
                reg=dict(type='SmoothL1Loss', beta=1.0, reduction='mean', loss_weight=1.0)),
            metrics=[
                dict(type='IOU', aggregate='none')
            ],
        )
    )
)
# dataset
dataset_type = 'LungDetPairDataset'
import os
data_root = os.getenv("MedDATASETS") + '/Detection/Luna2016/'
train_pipeline = [
    LoadImageFromFile(),
    LoadAnnotations(with_det=True),
    AnnotationMap({2: 1}),
    # LoadWeights(),
    Normalize(mean=(-400,), std=(700,), clip=True),
    Pad(size=(96, 96, 96)),
    # OneOf([
    #     WeightedCrop(patch_size=(96, 96, 96)),
    #     FirstDetCrop(patch_size=(96, 96, 96)),
    # ]),
    FirstDetCrop(patch_size=(96, 96, 96)),
    # RandomFlip(p=0.95),
    # RandomRotate(p=0.95, angle=15),
    # RandomScale(p=0.95, factor=0.15),
    # RandomShift(p=0.95, shift=0.05),
    # RandomGamma(p=0.95, gamma=0.2),
    # OneOf([
    #     RandomElasticDeformation(p=0.5),
    #     RandomBlur(p=0.95, sigma=1.5),
    # ]),
    # RandomNoise(p=0.95, std=0.025),
    # RandomBiasField(p=0.95, coefficients=0.025),
    Collect(keys=['img', 'gt_det']),
]
infer_pipeline = [
    LoadImageFromFile(),
    LoadPredictions(with_det=True),
    # Normalize(mean=(128,), std=(128,), clip=True),  # pay attention to the normalization
    # Normalize(mean=(300/900,), std=(1/900,), clip=False),  # pay attention to the normalization
    Normalize(mean=(-400,), std=(700,), clip=True),
    Pad(size=(96, 96, 96)),
    Patches(patch_size=(96, 96, 96)),
]
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        task=TASK,
        dataset_file=data_root + 'train_dataset_9.json',
        image_prefix=data_root,
        pipeline=train_pipeline),
    valid=dict(
        type=dataset_type,
        task=TASK,
        dataset_file=data_root + 'valid_dataset_9.json',
        image_prefix=data_root,
        pipeline=train_pipeline),
    infer=dict(
        type=dataset_type,
        task=TASK,
        dataset_file=data_root + 'infer_dataset_9.json',
        image_prefix=data_root,
        pipeline=infer_pipeline
    ),
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1.0 / 3,
    warmup_by_epoch=True,
    step=[75, 120])
checkpoint_config = dict(interval=10, save_optimizer=False)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
augmentation_cfg = dict(
    policy='poly',
    power=2,
    start=0.2,
)
total_epochs = 150
dist_params = dict(backend='nccl')
log_level = 'DEBUG'
work_dir = './work_dirs'
load_from = None
resume_from = None
workflow = [('train', 10), ('valid', 1)]
