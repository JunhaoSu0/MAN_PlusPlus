"""Stateful DDP MAN++ alias for ImageNet ResNet InfoPro.

This module exposes the MAN++ name while reusing the backward-compatible
stateful DDP implementation.
"""

from .resnetInfoPro_MAN_ddp import *  # noqa: F401,F403
