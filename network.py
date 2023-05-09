from typing import List, Union, cast, Dict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict

from prior_box import PriorBoxGenerator

def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)

class SSDFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.vgg16().features
        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d))

        # Patch ceil_mode for maxpool3 to get the same WxH output sizes as the paper
        backbone[maxpool3_pos].ceil_mode = True

        # parameters used for L2 regularization + rescaling
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # Multiple Feature maps - page 4, Fig 2 of SSD paper
        self.features = nn.Sequential(*backbone[:maxpool4_pos])  # until conv4_3

        # SSD300 case - page 4, Fig 2 of SSD paper
        extra = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        _xavier_init(extra)

        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True),
        )
        _xavier_init(fc)
        extra.insert(
            0,
            nn.Sequential(
                *backbone[maxpool4_pos:-1],  # until conv5_3, skip maxpool5
                fc,
            ),
        )
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # L2 regularization + Rescaling of 1st block's feature map
        x = self.features(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled]

        # Calculating Feature maps for the rest blocks
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


class SSDHead(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        # handle confidence loss
        self.conf_head = nn.ModuleList([
            nn.Conv2d(512, num_classes * 4, 3, padding=1),
            nn.Conv2d(1024, num_classes * 6, 3, padding=1),
            nn.Conv2d(512, num_classes * 6, 3, padding=1),
            nn.Conv2d(256, num_classes * 6, 3, padding=1),
            nn.Conv2d(256, num_classes * 4, 3, padding=1),
            nn.Conv2d(256, num_classes * 4, 3, padding=1),
        ])
        _xavier_init(self.conf_head)
        # handle localization loss
        self.loc_head = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, 3, padding=1),
            nn.Conv2d(1024, 4 * 6, 3, padding=1),
            nn.Conv2d(512, 4 * 6, 3, padding=1),
            nn.Conv2d(256, 4 * 6, 3, padding=1),
            nn.Conv2d(256, 4 * 4, 3, padding=1),
            nn.Conv2d(256, 4 * 4, 3, padding=1),
        ])
        _xavier_init(self.loc_head)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "conf_head_output": self.conf_head(x),
            "loc_head_output": self.loc_head(x),
        }

class SSD(nn.Module):
    def __init__(
        self,
        num_classes: int=21
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = SSDFeatureExtractor()
        self.head = SSDHead()

        self.prior_box_generator = PriorBoxGenerator()
        self.prior_boxes = self.prior_box_generator()
        print(self.prior_boxes)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def compute_loss(
        self,
        targets: List[Tensor], # Tensor size = [N, 5] (xmin, ymin, xmax, ymax, class_label)
        head_output: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ):
        conf_head_output = head_output["conf_head_output"]
        loc_head_output = head_output["loc_head_output"]

        for (
            targets_per_image,
            conf_per_image,
            loc_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, conf_head_output, loc_head_output, anchors, matched_idxs):
            # produce the matching between boxes and targets
            high_quality_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            high_quality_matched_idxs_per_image = matched_idxs_per_image[high_quality_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

print(SSD())