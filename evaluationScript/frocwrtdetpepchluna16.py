import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from noduleCADEvaluationLUNA16 import noduleCADEvaluation
import os
import csv
from multiprocessing import Pool
import functools
import SimpleITK as sitk
from glob import glob
import pandas as pd

SUFFIX = 'coord'
FOLD = 9

annotations_filename = './annotations/annotations.csv'  # path for ground truth annotations for the fold
annotations_excluded_filename = './annotations/annotations_excluded.csv'  # path for excluded annotations for the fold

output_dir = './folds_{0}/{1}'.format(SUFFIX, FOLD)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
seriesuids_filename = './folds_{0}/{1}/fold{1}_seriesuids.csv'.format(SUFFIX, FOLD)  # path for seriesuid for the fold
results_filename = './folds_{0}/{1}/fold{1}.csv'.format(SUFFIX, FOLD)


results_path = '../detector/results/res18/{}_{}/bbox/'.format(SUFFIX, FOLD)
sideinfopath = '/media/timothy/Datasets/Detection/DeepLungDataset/subset' + str(FOLD) + '/'
datapath = '/media/timothy/Disk1/LUNA2016/subset' + str(FOLD) + '/'

maxeps = 150  # 03 #150 #100#100
eps = [maxeps]  # range(1, maxeps + 1, 1)
detp = [-1.5, -1]  # , -0.5, 0]#, 0.5, 1]#, 0.5, 1] #range(-1, 0, 1)
isvis = False  # True
nmsthresh = 0.1
nprocess = 1  # 4
use_softnms = False
frocarr = np.zeros((maxeps, len(detp)))
firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']


def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip


def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


def nms(output, nms_th):
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


def convertcsv(bboxfname, bboxpath, detp):
    uid = bboxfname.split('_')[0]
    sliceim, origin, spacing, isflip = load_itk_image(datapath + '/' + uid + '.mhd')
    origin = np.load(sideinfopath + uid + '_origin.npy', mmap_mode='r')
    spacing = np.load(sideinfopath + uid + '_spacing.npy', mmap_mode='r')
    resolution = np.array([1, 1, 1])
    extendbox = np.load(sideinfopath + uid + '_extendbox.npy', mmap_mode='r')
    pbb = np.load(bboxpath + '/' + bboxfname, mmap_mode='r')

    pbbold = np.array(pbb[pbb[:, 0] > detp])
    pbbold = np.array(pbbold[pbbold[:, -1] > 3])  # add new 9 15
    # print(pbbold.shape)
    # pbb = np.array(pbb[:K, :4])
    # print pbbold.shape1
    # if use_softnms:
    #     keep = cpu_soft_nms(pbbold, method=2) # 1 for linear weighting, 2 for gaussian weighting
    #     pbb = np.array(pbbold[keep]) #cpu_soft_nms(pbbold)
    # else:
    pbb = nms(pbbold, nmsthresh)
    # print len(pbb), pbb[0]
    # print bboxfname, pbbold.shape, pbb.shape, pbbold.shape
    pbb = np.array(pbb[:, :-1])
    # print pbb[:, 0]
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:, 0], 1).T)
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)
    if isflip:
        Mask = np.load(sideinfopath + uid + '_mask.npy', mmap_mode='r')
        pbb[:, 2] = Mask.shape[1] - pbb[:, 2]
        pbb[:, 3] = Mask.shape[2] - pbb[:, 3]
    pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)
    rowlist = []
    # print pos.shape
    for nk in xrange(pos.shape[0]):  # pos[nk, 2], pos[nk, 1], pos[nk, 0]
        rowlist.append([uid, pos[nk, 2], pos[nk, 1], pos[nk, 0], 1 / (1 + np.exp(-pbb[nk, 0]))])
    # print len(rowlist), len(rowlist[0])
    return rowlist  # uid, pos[:K, 2], pos[:K, 1], pos[:K, 0], 1/(1+np.exp(-pbb[:K,0]))


# def getfrocvalue(results_filename, output_dir):
#     return noduleCADEvaluation(annotations_filename, annotations_excluded_filename, seriesuids_filename, results_filename, output_dir)
#
#
# def getcsv(detp, eps):
#     for ep in eps:
#         bboxpath = results_path + str(ep) + '/'
#         for detpthresh in detp:
#             print 'ep', ep, 'detp', detpthresh
#             f = open(bboxpath + 'predanno' + str(detpthresh) + 'd3.csv', 'w')
#             fwriter = csv.writer(f)
#             fwriter.writerow(firstline)
#             fnamelist = []
#             for fname in os.listdir(bboxpath):
#                 if fname.endswith('_pbb.npy'):
#                     fnamelist.append(fname)
#                     # print fname
#                     # for row in convertcsv(fname, bboxpath, k):
#                     # fwriter.writerow(row)
#             # # return
#             print(len(fnamelist))
#             predannolist = p.map(functools.partial(convertcsv, bboxpath=bboxpath, detp=detpthresh), fnamelist)
#             # print len(predannolist), len(predannolist[0])
#             for predanno in predannolist:
#                 # print predanno
#                 for row in predanno:
#                     # print row
#                     fwriter.writerow(row)
#             f.close()
#
#
# getcsv(detp, eps)
# def getfroc(detp, eps):
#     maxfroc = 0
#     maxep = 0
#     for ep in eps:
#         bboxpath = results_path + str(ep) + '/'
#         predannofnamalist = []
#         for detpthresh in detp:
#             predannofnamalist.append(bboxpath + 'predanno' + str(detpthresh) + '.csv')
#         froclist = p.map(getfrocvalue, predannofnamalist)
#         if maxfroc < max(froclist):
#             maxep = ep
#             maxfroc = max(froclist)
#         print froclist
#         for detpthresh in detp:
#             # print len(froclist), int((detpthresh-detp[0])/(detp[1]-detp[0]))
#             frocarr[(ep - eps[0]) / (eps[1] - eps[0]), int((detpthresh - detp[0]) / (detp[1] - detp[0]))] = \
#                 froclist[int((detpthresh - detp[0]) / (detp[1] - detp[0]))]
#             print 'ep', ep, 'detp', detpthresh, froclist[int((detpthresh - detp[0]) / (detp[1] - detp[0]))]
#     print maxfroc, maxep
#
#
# getfroc(detp, eps)
# p.close()
# fig = plt.imshow(frocarr.T)
# plt.colorbar()
# plt.xlabel('# Epochs')
# plt.ylabel('# Detection Prob.')
# xtick = detp  # [36, 37, 38, 39, 40]
# plt.yticks(range(len(xtick)), xtick)
# ytick = eps  # range(51, maxeps+1, 2)
# plt.xticks(range(len(ytick)), ytick)
# plt.title('Average FROC')
# plt.savefig(results_path + 'frocavg.png')
# np.save(results_path + 'frocavg.npy', frocarr)
# frocarr = np.load(results_path + 'frocavg.npy', 'r')
# froc, x, y = 0, 0, 0
# for i in xrange(frocarr.shape[0]):
#     for j in xrange(frocarr.shape[1]):
#         if froc < frocarr[i, j]:
#             froc, x, y = frocarr[i, j], i, j
# print fold, froc, x, y
# # print maxfroc


df = pd.DataFrame(columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
seriesuids = []
for i, filename in enumerate(sorted(glob(results_path + '/*_pbb.npy'))):
    print i, filename
    uid = os.path.basename(filename).split('_')[0]
    seriesuids.append(uid)

    a = convertcsv(
        bboxfname=os.path.basename(filename),
        bboxpath=results_path,
        detp=detp[1])
    for item in a:
        # print item
        df = df.append({'seriesuid':   item[0],
                        'coordX':      item[1],
                        'coordY':      item[2],
                        'coordZ':      item[3],
                        'probability': item[4]},
                       ignore_index=True)
    # break

pd.DataFrame(data=seriesuids).to_csv(seriesuids_filename, index=None, header=None)
df.to_csv(results_filename, index=None)

# p = Pool(16)
# fnamelist = [os.path.basename(f) for f in sorted(glob(results_path + '/*_pbb.npy'))]
# seriesuids = [i.split('_')[0] for i in fnamelist]
# predannolist = p.map(functools.partial(convertcsv, bboxpath=results_path, detp=detp[1]), fnamelist)
# df = pd.DataFrame(columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
# for predanno in predannolist:
#     # print item
#     for item in predanno:
#         df = df.append({'seriesuid':   item[0],
#                         'coordX':      item[1],
#                         'coordY':      item[2],
#                         'coordZ':      item[3],
#                         'probability': item[4]},
#                        ignore_index=True)
# pd.DataFrame(data=seriesuids).to_csv(seriesuids_filename, index=None, header=None)
# df.to_csv(results_filename, index=None)

froc = noduleCADEvaluation(annotations_filename, annotations_excluded_filename, seriesuids_filename, results_filename, output_dir)