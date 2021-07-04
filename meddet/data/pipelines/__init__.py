

# TODO solve big image transform
from .loading import LoadPrepare, \
    LoadImageFromFile, LoadAnnotations, LoadCoordinate, \
    LoadWeights, \
    LoadPseudoAsSeg, LoadSegAsImg, \
    LoadPredictions
from .saving import SaveImageToFile, SaveAnnotations, SaveFolder, SplitPatches
from .aug_intensity import RGB2Gray, Gray2RGB, ChannelSelect,\
    AnnotationMap, \
    Normalize, MultiNormalize, AutoNormalize, \
    ImageErosion, ImageDilation, ImageClosing, ImageOpening, \
    RandomBlur, RandomGamma, RandomNoise, RandomSpike, RandomBiasField, \
    RandomCutout, ForegroundCutout, \
    CLAHE, \
    LoadCannyEdge, LoadSkeleton # LoadGradient
from .aug_spatial import Resize, Pad, \
    RandomScale, RandomRotate, RandomShift, RandomFlip, \
    RandomElasticDeformation, \
    RandomCrop, WeightedCrop, CenterCrop, \
    ForegroundCrop, FirstDetCrop, OnlyFirstDetCrop, \
    ForegroundPatches
from .aug_other import Repeat
from .aug_testtime import Patches, MultiScale, MultiGamma
from .custom import MultiGammaEns
from .viewer import Viewer, Display
from .formating import Collect
from .compose import ForwardCompose, OneOf, BackwardCompose
from .statistic import Property
