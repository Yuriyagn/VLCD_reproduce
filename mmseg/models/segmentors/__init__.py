# Copyright (c) OpenMMLab. All rights reserved.
# 精简版 - 仅包含 ChangeCLIP 所需的分割器

from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from .encoder_decoderCD import EncoderDecoderCD
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .seg_tta import SegTTAModel
from .ChangeCLIPCD import ChangeCLIP
from .VLCD import VLCD

__all__ = [
    'BaseSegmentor', 'EncoderDecoder', 'EncoderDecoderCD',
    'CascadeEncoderDecoder', 'SegTTAModel', 'ChangeCLIP', 'VLCD'
]
