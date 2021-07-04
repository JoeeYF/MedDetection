from meddet.model.backbones.resnet import ResNet


class ResNeXt(ResNet):
    def __init__(self, groups=32, base_width=4, **kwargs):
        kwargs['groups'] = groups
        kwargs['base_width'] = base_width
        super(ResNeXt, self).__init__(**kwargs)


if __name__ == '__main__':
    import torch

    def init_seed(SEED):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    init_seed(666)

    r = ResNeXt(groups=32, base_width=4, dim=2, depth=50, downsample=1, in_channels=3)
    print(r)
    r.print_model_params()
    data = torch.ones((1, 3, 32, 32))
    outs = r(data)
    for o in outs:
        print(o.shape)
        print(torch.sum(o))