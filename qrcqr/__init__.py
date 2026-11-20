# -*- coding: utf-8 -*-
"""QRNN-CQR: target- and horizon-conditional spatial gating with a
recursive shared-head decoder and conformal exceedance warning."""

from .config import Config
from .model import QRNN

__all__ = ["Config", "QRNN"]
