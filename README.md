# Medical Image Detection
A detection framework for 3D medical images based on PyTorch refers to the modular design of [MMDetection](https://github.com/open-mmlab/mmdetection). 
The framework consists of one-stage and two-stage detectors. 
More methods, instructions, and experiment results will be updated.

## Installation
#### Requirements
- g++ 7.5
- gcc 7.5
- cuda > 10.1
- torch > 1.6.0

#### MedVision
```shell
git clone https://github.com/TimothyZero/MedVision.git
cd MedVision
pip install .
```

#### Train
```shell
git clone https://github.com/JoeeYF/MedDetection.git
cd MedDetection
python train.py --config config_path
bash run.sh config_name 1 1 1
```


##  Supported Methods
Detectors
- [x] Faster R-CNN
- [ ] Cascade R-CNN
- [x] RetinaNet
- [x] DeepLung
- [x] CenterNet
- [ ] yolov4

Backbone
- [x] ResNet
- [x] ResNeXt
- [x] SENet
- [ ] Res2Net

Neck
- [x] FPN
- [ ] PAN
- [ ] BiFPN

DenseHead
- [x] RetinaHead
- [x] RPNHead

RoIHead
- [x] BaseRoIHead
- [x] DoubleHead
- [ ] CascadeHead

Other features
- [x] OHEM
- [x] 3D DCNv2
- [ ] 3D Soft-NMS

## Dataset
### Luna
The annotation json files are in COCO format.
```
Detection/Luna2016
├── train_images_test
    ├── subset0
        ├── ....nii.gz
    ├── subset1
    ├── subset2
    ├── ...
    ├── subset9
├── infer_dataset_0.json
├── infer_dataset_1.json
├── infer_dataset_2.json
├── ...
├── infer_dataset_9.json
├── train_dataset_0.json
├── train_dataset_1.json
├── train_dataset_2.json
├── ...
└── train_dataset_9.json
```

## TODO
- [ ] more methods support
- [ ] experiment results
- [ ] dataset structure

## References

[MedVision](https://github.com/TimothyZero/MedVision)

[MMdetection](https://github.com/open-mmlab/mmdetection)

[medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
