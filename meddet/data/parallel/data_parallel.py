

from typing import Tuple, List
from itertools import chain
import torch
from torch.nn.parallel import DataParallel
from torch.nn.parallel._functions import Gather
from torch.nn.parallel.scatter_gather import scatter_kwargs


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if out is None:
            return None
        if isinstance(out, torch.Tensor):
            if out.ndim == 0:
                outputs = [o.unsqueeze(0) for o in outputs]
            try:
                return Gather.apply(target_device, dim, *outputs)
            except Exception as e:
                print(out.shape, len(outputs))
                raise e
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return dict([(k, gather_map([d[k] for d in outputs])) for k in out])
        if isinstance(out, (float, int)):
            return outputs
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


class MedDataParallel(DataParallel):
    """
    replicate: replicate a Module on multiple devices
    scatter: distribute the input in the first-dimension
    gather: gather and concatenate the input in the first-dimension
    parallel_apply: apply a set of already-distributed inputs to a set of already-distributed models.
    """

    def parse_losses(self, losses_dict):
        num_samples = losses_dict.pop('num_samples')
        SUM = torch.sum(num_samples)
        # for k, v in losses_dict.items():
        #     losses_dict[k] = torch.sum(v * num_samples) / SUM
        loss, log_vars = self.module.parse_losses(losses_dict)
        log_outputs = dict(loss=loss, log_vars=log_vars, num_samples=SUM.item())
        return log_outputs

    def scatter(self, inputs: tuple, kwargs: dict, device_ids: list) -> Tuple[Tuple[tuple], Tuple[dict]]:
        """
        scatter inputs to gpu ids
        Args:
            inputs: model inputs used in forward, e.g., (data_batch, optimizers)
            kwargs: model kwargs used in forward
            device_ids: used cpu devices

        Returns:
            inputs: tuple inputs for all gpus
            kwargs: tuple kwargs for all gpus
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs: tuple, output_device: list):
        return gather(outputs, output_device, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.train_iter(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)

        if len(self.device_ids) == 1:
            outputs = self.module(*inputs[0], **kwargs[0])
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, kwargs)
            outputs = self.gather(outputs, self.output_device)

        outputs = list(outputs)
        log_outputs = self.parse_losses(outputs.pop(0))
        return (log_outputs, *outputs)