import os
import os.path as osp
import shutil
import torch
import time
import random
import numpy as np
import argparse
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from meddet import setting
from meddet.utils import Config, get_root_logger
from meddet.task import builder, DetMonitor
from meddet.data import build_dataset, build_dataloader, MedDataParallel
from meddet.runner import build_optimizer, MedRunner


def init_seed(SEED, deterministic=True):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_task(cfg):
    assert cfg.TASK in ('SEG', 'CLS', 'DET')
    if isinstance(cfg.model, dict):
        cfg.model['dim'] = cfg.DIM
    if cfg.TASK.upper() == 'DET':
        model = builder.build_detector(cfg.model)
        monitor = DetMonitor(cfg.work_dir, 0.1)
    else:
        raise NotImplementedError
    return model, monitor


def build_loaders(cfg, args):
    if hasattr(cfg.data, 'all'):
        logger.info(f'------ training with {setting.K_FOLD} CV --------')
        all_dataset = build_dataset(cfg.data.all)
        if len(cfg.workflow) == 2:
            train_indices, valid_indices = all_dataset.cross_valid_split(setting.K_FOLD, args.fold)
            indices = [train_indices, valid_indices]
            logger.info(indices)
            logger.info(list(map(len, indices)))
        else:
            indices = [list(range(len(all_dataset)))]
            logger.info(indices)
            logger.info(list(map(len, indices)))

        data_loaders = [
            build_dataloader(all_dataset,
                             cfg.data.imgs_per_gpu,
                             cfg.data.workers_per_gpu,
                             cfg.gpus,
                             sampler=SubsetRandomSampler(_indices))
            for _indices in indices
        ]
    else:
        datasets = [build_dataset(cfg.data.train)]
        print(len(datasets[0]))
        if len(cfg.workflow) == 2:
            datasets.append(build_dataset(cfg.data.valid))
            print(len(datasets[1]))
        data_loaders = [
            build_dataloader(
                dataset,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                cfg.gpus,
                shuffle=True
            ) for dataset in datasets
        ]
    return data_loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/faster_rcnn_deeplung_coord.py')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=457)
    parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    init_seed(args.seed)
    print('init seed', args.seed, np.random.get_state()[1][0])


    cfg = Config.fromfile(args.config)
    if args.batch:
        cfg.data.imgs_per_gpu = args.batch
    if args.workers:
        cfg.data.workers_per_gpu = args.workers
    if args.gpus:
        cfg.gpus = args.gpus

    cfg.work_dir = osp.join(cfg.work_dir, cfg.module_name)

    os.makedirs(osp.join(cfg.work_dir, 'logs'), exist_ok=True)
    os.makedirs(osp.join(cfg.work_dir, 'figs'), exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'logs', '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    shutil.copy(cfg.filename, os.path.join(cfg.work_dir, timestamp + "_" + os.path.basename(cfg.filename)))

    data_loaders = build_loaders(cfg, args)
    model, monitor = build_task(cfg)
    model = MedDataParallel(model.cuda(0), device_ids=range(cfg.gpus))  # must put model on first gpu
    optimizer = build_optimizer(model, cfg.optimizer)
    # model.module.setLog()
    # print(model.module)
    # monitor.clearAll()

    # logger.debug(f'Setting:\n'
    #              f'\tConv: {setting.CONV_CFG}\n'
    #              f'\tAct: {setting.ACT_CFG}\n'
    #              f'\tNorm: {setting.NORM_CFG}')
    logger.debug('Config:\n{}'.format(cfg.pretty_text))
    logger.debug('Model:\n{}'.format(model.module.__repr__()))

    runner = MedRunner(
        model,
        monitor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        timestamp=timestamp,
        float16=cfg.FLOAT16)

    runner.register_training_hooks(cfg.lr_config,
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   cfg.augmentation_cfg)
    if args.resume:
        runner.resume(cfg.work_dir + f'/epoch_{args.resume}.pth')
    else:
        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
