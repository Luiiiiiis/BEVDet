# Copyright (c) Phigent Robotics. All rights reserved.
"""BEVDet migration project for mmdet3d 1.4.

All modules are registered into ``mmdet3d``'s registries (``MODELS`` /
``TRANSFORMS``) at import time, so a config can directly use::

    custom_imports = dict(imports=['bevdet'], allow_failed_imports=False)
"""
from . import models  # noqa: F401  register models
from . import datasets  # noqa: F401  register transforms

__all__ = ['models', 'datasets']
