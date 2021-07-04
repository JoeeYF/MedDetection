 

# -*- coding:utf-8 -*-

import torch.nn as nn
from meddet.model.heads.dense_heads.anchor_head import AnchorHead, images_to_levels



class DeepLungHead(AnchorHead):

    def __init__(self, **kwargs):
        super(DeepLungHead, self).__init__(**kwargs)
        self.init_weights()

    def _init_layers(self):
        self.output = nn.Sequential(nn.Conv3d(128, 64, kernel_size=1),
                                    nn.ReLU(),
                                    # nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(64, 7 * self.num_anchors, kernel_size=1))

    def init_weights(self):
        pass
        # for m in self.modules():
        #     if self.is_conv(self.dim, m):
        #         n = np.prod(m.kernel_size) * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif self.is_norm(self.dim, m):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #
        # prior = 0.01
        # self.rpn_cls.weight.data.fill_(0)
        # self.rpn_cls.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #
        # self.rpn_reg.weight.data.fill_(0)
        # self.rpn_reg.bias.data.fill_(0)

        # normal_init(self.rpn_bone, std=0.01)
        # normal_init(self.rpn_cls, std=0.001, bias=-math.log((1.0 - prior) / prior))
        # normal_init(self.rpn_reg, std=0.01)

    def forward_single_level(self, single_level_feature):
        out = self.output(single_level_feature)
        cls_out = out[:, :self.anchor_generator.num_base_anchors, ...]
        reg_out = out[:, self.anchor_generator.num_base_anchors:, ...]
        return cls_out, reg_out