

from .aug_base import OperationStage
from ..registry import PIPELINES


@PIPELINES.register_module
class Repeat(OperationStage):
    def __init__(self, times):
        super().__init__()
        self.times = times

    @property
    def repeats(self):
        return self.times
