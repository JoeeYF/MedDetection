

from abc import ABC

from ..task import BaseTask


class BaseDetector(BaseTask, ABC):
    def __init__(self):
        super().__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def extract_feat(self, imgs):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(imgs)
        if self.with_neck:
            x = self.neck(x)
        return x