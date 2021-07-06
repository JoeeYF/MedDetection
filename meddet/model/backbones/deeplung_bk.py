import torch
from torch import nn
from medvision.ops import DeformConv3dPack as DCNv1
from medvision.ops import ModulatedDeformConv3dPack as DCNv2
from meddet.model.nnModules import ComponentModule


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1, dcn=None, coord=False):
        super(PostRes, self).__init__()
        self.coord = coord
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        if dcn is not None:
            dcn_ = dcn.copy()
            layer_type = dcn_.pop('type')
            self.conv2 = eval(layer_type)(n_out, n_out, kernel_size=3, stride=1, padding=1, **dcn_)
        else:
            self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class DeepLungBK(ComponentModule):
    def __init__(self,
                 dim,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 coord=False,
                 ):
        super(DeepLungBK, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.dcn = dcn
        self.stage_with_dcn=stage_with_dcn
        self.dim = dim
        self.coord = coord
        self.anchors = [5., 10., 20.]
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        if self.dcn is not None:
            assert len(stage_with_dcn) == len(num_blocks_forw)
        for i in range(len(num_blocks_forw)):
            blocks = []
            dcn = self.dcn if self.stage_with_dcn[i] else None
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1], dcn=dcn))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1], dcn=dcn))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    addition = 3 if self.coord and i == 0 else 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpoolcoord1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=False)
        self.maxpoolcoord2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=False)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, img, coord=None):
        x = img
        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        # out2 = self.drop(out2)
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 96
        # out4 = self.drop(out4)

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))  # 96+96
        # comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)

        if self.coord:
            coord = self.maxpoolcoord2(self.maxpoolcoord1(coord))
            comb2 = self.back2(torch.cat((rev2, out2, coord), 1))  # 64+64
        else:
            comb2 = self.back2(torch.cat((rev2, out2), 1))  # 64+64
        comb2 = self.drop(comb2)
        # comb2 = self.drop(comb2)
        # out_cls = self.output_cls(comb2)
        # out_reg = self.output_reg(comb2)
        #
        # return [out_cls], [out_reg]
        return [comb2]


if __name__ == "__main__":
    net = DeepLungBK(3, dict(type='DeformConv3dPack', deformable_groups=1), stage_with_dcn=(False, False, False, False)).cuda()
    x = torch.rand((1, 1, 96, 96, 96)).half().cuda()
    data = {
        'img': x,
        'img_meta': None,
        'gt_labels': 2 * torch.rand((1, 24, 24, 24, 3, 5)) - 1,
        'gt_bboxes': torch.ones((1, 3, 24, 24, 24)),
    }
    multi_level_cls_out = net(data['img'])
    [print(i.shape) for i in multi_level_cls_out]
    # [print(i.shape) for i in multi_level_reg_out]
