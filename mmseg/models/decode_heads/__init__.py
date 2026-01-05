# Copyright (c) OpenMMLab. All rights reserved.
# 精简版 - 仅包含 ChangeCLIP 所需的解码头

from .decode_head import BaseDecodeHead
from .fpn_head import FPNHead
from .swin_text_head import SwinTextDecode
from .denseclip_heads import IdentityHead

__all__ = [
    'BaseDecodeHead', 'FPNHead', 'SwinTextDecode', 'IdentityHead'
]
