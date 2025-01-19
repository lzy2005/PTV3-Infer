num_classes = 13
backbone_out_channels = 64
criteria = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
]

import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

class Segmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.backbone = PointTransformerV3()
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)

        return dict(seg_logits=seg_logits)