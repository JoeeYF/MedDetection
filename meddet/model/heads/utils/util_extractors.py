 

import warnings
import torch
from torch import nn
from abc import ABCMeta, abstractmethod
try:
    from medvision.ops import RoIAlign
except ImportError:
    warnings.warn('Using torchvision.ops.RoIAlign instead.')
    from torchvision.ops import RoIAlign


from meddet.model.nnModules import BlockModule


class BaseRoIExtractor(BlockModule):
    """Base class for RoI extractor.f
    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (list): Strides of input feature maps.
    """

    def __init__(self, dim, roi_layer, out_channels, featmap_strides):
        super(BaseRoIExtractor, self).__init__()
        self.dim = dim
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.output_size = self.roi_layers[0].output_size

    @property
    def num_inputs(self):
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.
        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (list): The stride of input feature map w.r.t to the
                original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.
        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        if isinstance(cfg['output_size'], int):
            cfg['output_size'] = (cfg['output_size'], ) * self.dim
        elif type(cfg['output_size']) in [list, tuple]:
            assert len(cfg['output_size']) == self.dim
        layer_cls = eval(layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    @staticmethod
    def roi_rescale(rois, scale_factor):
        """Scale RoI coordinates by scale factor.
        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 1 + 2 * dim)
            scale_factor (float): Scale factor that RoI will be multiplied by.
        Returns:
            torch.Tensor: Scaled RoI.
        """
        dim = rois.shape[-1] // 2
        shape = rois[:, dim+1:] - rois[:, 1:dim+1]
        new_rois = rois.clone()
        new_rois[:, dim+1:] = new_rois[:, dim+1:] + (scale_factor - 1.) * shape / 2
        new_rois[:, 1:dim+1] = new_rois[:, 1:dim+1] - (scale_factor - 1.) * shape / 2
        return new_rois

    @abstractmethod
    def forward(self, feats, rois, roi_scale_factor=None):
        pass


class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.
    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.
    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (list): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 finest_scale=56,
                 **kwargs):
        self.finest_scale = finest_scale
        super(SingleRoIExtractor, self).__init__(**kwargs)

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.
        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3
        Args:
            rois (Tensor): Input RoIs, shape (k, 5/7).
            num_levels (int): Total level number.
        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        dim = rois.shape[1] // 2
        scale = torch.pow(torch.prod(rois[:, dim + 1: 2 * dim + 1] - rois[:, 1: dim + 1], dim=1), 1 / dim)
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats: list, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            # TODO: make it nicer when exporting to onnx
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats


class GenericRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from all level feature maps levels.
    This is the implementation of `A novel Region of Interest Extraction Layer
    for Instance Segmentation <https://arxiv.org/abs/2004.13665>`_.
    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Default: 'sum'.
        pre_cfg (dict | None): Specify pre-processing modules. Default: None.
        post_cfg (dict | None): Specify post-processing modules. Default: None.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    """

    def __init__(self,
                 aggregation='mean',
                 **kwargs):
        super(GenericRoIExtractor, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat', 'mean']

        self.aggregation = aggregation

    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # mark the starting channels for concat mode
        start_channels = 0
        for i in range(num_levels):
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            end_channels = start_channels + roi_feats_t.size(1)
            if self.aggregation in ['sum', 'mean']:
                # and sum them all
                roi_feats += roi_feats_t
            else:
                # and concat them along channel dimension
                roi_feats[:, start_channels:end_channels] = roi_feats_t
            # update channels starting position
            start_channels = end_channels
        # check if concat channels match at the end
        if self.aggregation == 'concat':
            assert start_channels == self.out_channels
        elif self.aggregation == 'mean':
            roi_feats /= num_levels

        return roi_feats


if __name__ == '__main__':
    a = SingleRoIExtractor(
        dim=3,
        roi_layer=dict(type='RoIAlign', output_size=(7, 7, 7), sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32])
    print(a.output_size)