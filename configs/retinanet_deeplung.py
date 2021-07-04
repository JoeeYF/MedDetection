
from meddet.task import *
from meddet.data.pipelines import *

DIM = 3
TASK = 'DET'
CLASSES = 1
FLOAT16 = True
model = RetinaNet(
    dim=DIM,
    backbone=DeepLungBK(
        dim=DIM
    ),
    neck=None,
    head=RetinaHead(
        dim=DIM,
        in_channels=128,
        feat_channels=128,
        num_classes=CLASSES,
        level_first=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            base_scales=1,
            scales=[1.25, 2.5, 5],
            ratios=[1],
            strides=[4]
        ),
        bbox_coder=dict(
            type='DeltaBBoxCoder',
            target_stds=[1., 1., 1., 1., 1., 1.]),
        assigner=dict(
            type='IoUAssigner',
            pos_iou_thr=0.7 ** DIM,
            neg_iou_thr=0.2 ** DIM,
            min_pos_iou=0.5 ** DIM,
            match_low_quality=True,
            num_neg=800),
        sampler=dict(
            type='HNEMSampler',
            pos_fraction=0.35),
        nms=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.15,
            nms_fun=dict(type='nms', iou_threshold=0.1),
            max_per_img=100),
        losses=dict(
            cls=CrossEntropyLoss(use_sigmoid=True),
            reg=SmoothL1Loss(beta=1.0, reduction='mean', loss_weight=1.0)),
        metrics=[
            IOU(aggregate='none')
        ],
    )
)
# dataset
dataset_type = 'LungDetPairDataset'
data_root = '/home/qiao/Desktop/Detection/Luna2016/'
train_pipeline = [
    LoadImageFromFile(),
    LoadAnnotations(with_det=True),
    AnnotationMap({2: 1}),
    Normalize(mean=(-400,), std=(700,), clip=True),
    Pad(size=(96, 96, 96)),
    FirstDetCrop(patch_size=(96, 96, 96)),
    Collect(keys=['img', 'gt_det']),
]
infer_pipeline = [
    LoadImageFromFile(),
    LoadPredictions(with_det=True),
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
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup=None,
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
# resume_from = "./work_dirs/cfg_sub_model_3_nodule_det_retina_v3/epoch_20.pth"
resume_from = None
workflow = [('train', 10), ('valid', 1)]
