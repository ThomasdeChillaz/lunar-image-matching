"""Explicit larger-resolution EfficientLoFTR variants on the same native pixels.

Higher internal resolution adds no source information. It tests the interaction
between pretrained receptive fields and tiny 128px lunar crops. Native-grid
pixel-center conversion and author source/weights remain unchanged.
"""
from eloftr_matcher import EfficientLoFTRMatcher


class ScaledEfficientLoFTRMatcher(EfficientLoFTRMatcher):
    def __init__(self, dependency_dir=None, factor=2, device='mps'):
        if factor not in (2, 4):
            raise ValueError('This separately declared experiment supports 2x/4x only')
        super().__init__(dependency_dir=dependency_dir, factor=2, device=device)
        # Base class's image conversion is scale-generic; keep its native/2x
        # public experiment unchanged while declaring 4x in this separate method.
        self.factor = factor
