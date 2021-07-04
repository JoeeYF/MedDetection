import inspect
import warnings


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        # print(name)

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module_class (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            print('\033[31m{} is already registered in {}\033[0m'.format(module_name, self.name))
            return
        self._module_dict[module_name] = module_class
        # print(module_name)

    def register_module(self, cls):
        self._register_module(cls)
        return cls

    def register_modules(self, classes):
        [self._register_module(cls) for cls in classes]
        return None

    def build_from_cfg(self, cfg, default_args=None):
        """Build a module from config dict.
        registry (:obj:`Registry`): The registry to search the type from.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            default_args (dict, optional): Default initialization arguments.
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and 'type' in cfg
        assert isinstance(default_args, dict) or default_args is None
        args = cfg.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError('{} is not in the {} registry'.format(obj_type, self.name))
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
        try:
            return obj_cls(**args)
        except Exception as e:
            raise Exception(f"error while build class: {obj_type}\n" + str(e) + "\n" + str(args))


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    if callable(cfg):
        return cfg
    if cfg is None:
        return None
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))
    
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    try:
        return obj_cls(**args)
    except Exception as e:
        raise Exception(f"error while build class: {obj_type}\n" + str(e) + "\n" + str(args))
