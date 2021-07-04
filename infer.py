

import os.path as osp
import torch
import argparse
import numpy as np

from meddet.utils import Config, get_root_logger
from meddet.task import builder, DetMonitor
from meddet.data import build_dataset, build_dataloader, Viewer, ImageIO
from meddet.data.pipelines import *


d = Display()

def merge_seg_pred(pred_results: list, method='mean'):
    pred_results = np.stack(pred_results, axis=0)
    if method == 'mean':
        pred_results = np.mean(pred_results, axis=0) > 0.5
    elif method == 'max':
        pred_results = np.max(pred_results, axis=0)
    return pred_results.astype(np.float32)


def merge_det_pred(pred_results: list, method='nms'):
    pass


def det_infer_net(net, dataset):
    v = Viewer(monitor.result_dir, 1.0)

    net.eval()
    for i, data in enumerate(dataset):
        if isinstance(data, dict):
            data = [data]

        pred_results = []
        for multi_idx, results in enumerate(data):
            filename = results['filename']

            results['seg_fields'] = []
            dim = results['img_dim']

            if 'patches_img' in results.keys():
                for p, image in enumerate(results['patches_img']):
                    tensor_data = {'img': torch.from_numpy(image).unsqueeze(0).cuda(),
                                   'img_meta': None}
                    if 'patches_gt_coord' in results:
                        coord = results.pop('patches_gt_coord')
                        tensor_data['gt_coord'] = torch.from_numpy(coord[p]).unsqueeze(0).cuda()
                    with torch.no_grad():
                        prediction, _ = model.forward_infer(tensor_data)
                    pred_bboxes = prediction.cpu().numpy()[0]  # batch size = 1
                    pred_bboxes = pred_bboxes[pred_bboxes[:, -1] != -1]
                    results['patches_pred_det'][p] = pred_bboxes

                results = monitor.result_pipeline(results)
                results = saver(results)
                results['gt_det'] = results['pred_det']
                print(len(results['pred_det']))
                print(results['pred_det'])
            else:
                tensor_data = {'img': torch.from_numpy(results['img']).unsqueeze(0).cuda(),
                               'img_meta': None}
                with torch.no_grad():
                    prediction, _ = model.forward_infer(tensor_data)
                pred_bboxes = prediction.cpu().numpy()[0]
                pred_bboxes = pred_bboxes[pred_bboxes[:, -1] != -1]
                results['pred_det'] = pred_bboxes

                results = monitor.result_pipeline(results)
                results = saver(results)
                results['gt_det'] = results['pred_det']
                print(len(results['pred_det']))
                print(results['pred_det'])

            # v(results)
        #         pred_results.append(results['pred_seg'])
        #         ImageIO.saveArray(os.path.join(monitor.result_dir, f"{filename}_{multi_idx}_pred_seg.{ext}"),
        #                           results['pred_seg'], spacing=results['img_spacing'], origin=results['img_origin'])
        #
        # pred_results = merge_seg_pred(pred_results, method='mean')
        # ImageIO.saveArray(os.path.join(monitor.result_dir, f"{filename}_pred_seg.{ext}"),
        #                   pred_results, spacing=results['img_spacing'], origin=results['img_origin'])
        # ImageIO.saveArray(os.path.join(monitor.result_dir, f"{filename}_img.{ext}"),
        #                   results['img'], spacing=results['img_spacing'], origin=results['img_origin'])

        # if i >= 0:
        #     return


# def infer_net_loader(net, loader, batch_processor):
#     monitor.setInferMode()
#     v = ViewerND(None, 1.0)
#
#     net.eval()
#     for i, data in enumerate(loader):
#         batch_size = len(data['img'])
#         for j in range(batch_size):
#             results = {'img_meta': data['img_meta'][j]}
#             for k, v in data.items():
#                 if k != 'img_meta':
#                     results[k] = v[j]
#
#             print(results.keys())
#             print(results['img_meta'].keys())
#             filename = results['img_meta']['filename']
#             print(filename)
#             print(results['img'].shape, results['patches_img'].shape)
#             dim = results['img_meta']['img_dim']
#             if dim == 3:
#                 ext = 'nii.gz'
#             else:
#                 ext = 'png'
#             if 'patches_img' in results.keys():
#                 results['img_meta']['patches_pred_seg'] = np.zeros_like(results['patches_img'])
#                 results['img_meta']['pred_seg'] = np.zeros_like(results['img'][[0], ...])
#                 results['img_meta']['seg_fields'].append('pred_seg')
#
#                 for p in range(results['patches_img'].shape[0]):
#
#                     tensor_data = {'img': results['patches_img'][[p], ...],
#                                    'img_meta': None}
#                     print(tensor_data['img'].shape)
#                     with torch.no_grad():
#                         net_output = batch_processor(model, tensor_data)
#                     segmentation = np.argmax(net_output.squeeze(0).cpu().numpy(), axis=0).astype(np.float32)
#                     print(results['img_meta']['patches_pred_seg'][p, ...].shape, segmentation.shape)
#                     results['img_meta']['patches_pred_seg'][p, ...] = segmentation[None, ...]
#
#                 axes = [0] + list(range(2, 2 + dim)) + [1]
#                 results['img_meta']['patches_pred_seg'] = results['img_meta']['patches_pred_seg'].transpose(axes)
#                 # axes = list(range(1, 1 + dim)) + [0]
#                 # results['img_meta']['pred_seg'] = results['img_meta']['pred_seg'].transpose(axes)
#                 print(results['img_meta']['pred_seg'].shape, results['img_meta']['patches_pred_seg'].shape)
#
#                 results = monitor.result_pipeline(results)[0]
#                 results['gt_seg'] = results['pred_seg']
#                 # v(data)
#                 print(results['pred_seg'].shape, results['img'].shape)
#                 ImageIO.saveArray(os.path.join(monitor.result_dir, f"{filename}_pred_seg.{ext}"),
#                                   results['pred_seg'], spacing=results['img_spacing'], origin=results['img_origin'])
#                 ImageIO.saveArray(os.path.join(monitor.result_dir, f"{filename}_img.{ext}"),
#                                   results['img'], spacing=results['img_spacing'], origin=results['img_origin'])
#                 # monitor.saveResults(data, net_output, sp=1.0)
#                 # processor.setInferMode()
#                 # processor.viewFeatures(data, net_output, force_save=True)
#                 # processor.viewResults(data, net_output, force_save=True)
#                 # processor.saveResults(data, torch.from_numpy(data['gt_seg']), sp=1.0)
#
#         if i > 200:
#             break


def build_task(cfg):
    assert cfg.TASK in ('SEG', 'CLS', 'DET')
    if isinstance(cfg.model, dict):
        cfg.model['dim'] = cfg.DIM
    if cfg.TASK.upper() == 'SEG':
        model = builder.build_segmenter(cfg.model).to(device)
        monitor = SegMonitor(cfg.work_dir, 1.0, cfg.infer_pipeline[::-1])
        print(monitor.result_pipeline)
    elif cfg.TASK.upper() == 'CLS':
        model = builder.build_classifier(cfg.model).to(device)
        monitor = SegMonitor(cfg.work_dir, 1.0, cfg.infer_pipeline[::-1])
    elif cfg.TASK.upper() == 'DET':
        model = builder.build_detector(cfg.model).to(device)
        monitor = DetMonitor(cfg.work_dir, 1.0, cfg.infer_pipeline[::-1])
    else:
        raise NotImplementedError
    return model, monitor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('--ckpt', default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # args.config = 'configs/faster_rcnn_deeplung_coord.py'
    # args.ckpt = 'work_dirs/faster_rcnn_deeplung_coord/epoch_130.pth'


    cfg = Config.fromfile(args.config)
    checkpoint = args.ckpt

    print(cfg.filename)
    print(checkpoint)

    """==============================================="""
    cfg.gpus = 1
    cfg.work_dir = osp.join(cfg.work_dir, cfg.module_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, monitor = build_task(cfg)
    monitor.setInferMode()
    # model.setHook()

    cfg.data.infer.infer_mode = True
    dataset = build_dataset(cfg.data.infer)
    print('infer dataset', len(dataset))

    loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,  # must be 1
        workers_per_gpu=1,
        shuffle=False,
        drop_last=False
    )

    # put model on gpus
    # model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    model.load_state_dict(
        torch.load(checkpoint, map_location=device)["state_dict"])

    if cfg.TASK == 'DET':
        saver = ForwardCompose([
            SaveFolder(monitor.result_dir),
            # SaveImageToFile(ext='.nii.gz'),
            SaveAnnotations(with_det=True),
        ])
        print(saver)
        det_infer_net(model, dataset)
    elif cfg.TASK == 'SEG':
        saver = ForwardCompose([
            SaveFolder(monitor.result_dir),
            # SaveImageToFile(ext='same'),
            # SaveImageToFile(ext='.nii.gz'),
            SaveAnnotations(with_seg=True),
        ])
        print(saver)
        seg_infer_net(model, dataset)

    # infer_net_loader(model, loader, batch_processor)
