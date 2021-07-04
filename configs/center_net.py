from meddet.task import *
from meddet.data.pipelines import *

DIM = 3
TASK = 'DET'
CLASSES = 1
FLOAT16 = True
model = dict(
    type='CenterNet',
    dim=DIM,
    backbone=SCPMNet(
        dim=DIM,
        depth=18,
        in_channels=1,
        out_channels=64
    ),
    neck=None,
    head=dict(type='CenterNetHead',
              dim=DIM,
              in_channel=64,
              feat_channel=64,
              num_classes=CLASSES,
              loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
              loss_wh=dict(type='L1Loss', loss_weight=0.1),
              loss_offset=dict(type='L1Loss', loss_weight=1.0),
              test_cfg=dict(topk=20, local_maximum_kernel=3),
              nms=dict(
                  nms_pre=1000,
                  min_bbox_size=0,
                  score_thr=0.15,
                  nms_fun=dict(type='nms', iou_threshold=0.1),
                  max_per_img=100),
              metrics=[
                  IOU(aggregate='none'),
                  Dist(aggregate='none', dist_threshold=5)]
              ),
)

# dataset
dataset_type = 'LungDetPairDataset'
import os
data_root = os.getenv("MedDATASETS") + '/Detection/Luna2016/'
train_pipeline = [
    LoadImageFromFile(),
    LoadAnnotations(with_det=True),
    LoadCoordinate(),
    AnnotationMap({2: 1}),
    Normalize(mean=(-400,), std=(700,), clip=True),
    Pad(size=(96, 96, 96)),
    FirstDetCrop(patch_size=(96, 96, 96)),
    Collect(keys=['img', 'gt_det', 'gt_coord']),
]
infer_pipeline = [
    LoadImageFromFile(),
    LoadPredictions(with_det=True),
    LoadCoordinate(),
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
