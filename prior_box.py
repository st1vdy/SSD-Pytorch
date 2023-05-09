import math
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor

def prior_box_transform(boxes: Tensor, image_size: Tuple[int, int] = [300, 300]) -> Tensor:
    '''
    将(cx, cy, w, h)形式的bbox转换为(xmin, ymin, xmax, ymax)形式
    '''
    w = image_size[0]
    h = image_size[1]
    x_min = (boxes[:, 0] - boxes[:, 2] / 2) * w
    x_max = (boxes[:, 0] + boxes[:, 2] / 2) * w
    y_min = (boxes[:, 1] - boxes[:, 3] / 2) * h
    y_max = (boxes[:, 1] + boxes[:, 3] / 2) * h

    x_min = x_min.clamp(min=0, max=w - 1).round().int()
    x_max = x_max.clamp(min=0, max=w - 1).round().int()
    y_min = y_min.clamp(min=0, max=h - 1).round().int()
    y_max = y_max.clamp(min=0, max=h - 1).round().int()
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

def prior_box_inverse_transform(boxes: Tensor, image_size: Tuple[int, int] = [300, 300]) -> Tensor:
    '''
    将(xmin, ymin, xmax, ymax)形式的bbox转换为(cx, cy, w, h)形式
    '''
    w = image_size[0]
    h = image_size[1]
    c_x = (boxes[:, 0] + boxes[:, 2]) / (2 * w)
    c_y = (boxes[:, 1] + boxes[:, 3]) / (2 * h)
    lw = (boxes[:, 2] - boxes[:, 0]) / w
    lh = (boxes[:, 3] - boxes[:, 1]) / h

    return torch.stack([c_x, c_y, lw, lh], dim=1)

def box_encode(gt_boxes: Tensor, prior_boxes: Tensor, std: Tuple[float, float] = [0.1, 0.2], eps = 1e-6):
    '''
    编码预测框的偏移量
    '''
    gt_boxes = prior_box_inverse_transform(gt_boxes) # 转换为(cx, cy, w, h)形式
    print(gt_boxes)
    print(prior_boxes)
    shift_center = (gt_boxes[:, :2] - prior_boxes[:, :2]) / (std[0] * prior_boxes[:, 2:])
    shift_size = (torch.log(gt_boxes[:, 2:] / (prior_boxes[:, 2:] + eps)) + eps) / std[1] # eps是防止0
    return torch.concatenate([shift_center, shift_size], dim=1)

def box_decode(shifts: Tensor, prior_boxes: Tensor, std: Tuple[float, float] = [0.1, 0.2]):
    '''
    根据偏移量和预测框还原框的位置
    '''
    box_center = shifts[:, :2] * std[0] * prior_boxes[:, 2:] + prior_boxes[:, :2]
    box_size = torch.exp(shifts[:, 2:] * std[1]) * prior_boxes[:, 2:]
    return prior_box_transform(torch.concatenate([box_center, box_size], dim=1))

def prior_cox_crop(boxes: Tensor, image_size: Tuple[int, int] = [300, 300]) -> Tensor:
    boxes = prior_box_transform(boxes, image_size)
    boxes = prior_box_inverse_transform(boxes, image_size)
    return boxes

class PriorBoxGenerator:
    def __init__(
        self,
        s_min: float = 0.2,
        s_max: float = 0.9,
        image_size: Tuple[int, int] = [300, 300],
        aspect_ratios: List[List[int]] = [[2], [2,3], [2,3], [2,3], [2], [2]],
        feature_map_sizes: List[int] = [38, 19, 10, 5, 3, 1]
    ):
        assert len(feature_map_sizes) == len(aspect_ratios)

        scales = np.linspace(s_min, s_max, len(aspect_ratios))
        self.scales = np.concatenate((scales, [1.05]))
        print(self.scales)
        self.w = image_size[0] # width
        self.h = image_size[1] # height
        self.aspect_ratios = aspect_ratios
        self.feature_map_sizes = feature_map_sizes

    def __call__(self) -> Tensor:
        prior_boxes = []
        for i, f_k in enumerate(self.feature_map_sizes):
            c_x = (torch.arange(0, f_k) + 0.5) / f_k
            c_y = (torch.arange(0, f_k) + 0.5) / f_k
            c_x, c_y = torch.meshgrid(c_x, c_y, indexing='ij')
            c_x = c_x.reshape(-1)
            c_y = c_y.reshape(-1)
            fm_x = 1 / f_k
            fm_y = 1 / f_k

            s_k = torch.ones(c_x.shape[0]) * self.scales[i]
            s_k_prime = torch.ones(c_x.shape[0]) * math.sqrt(self.scales[i] * self.scales[i + 1])
            prior_boxes.append(torch.stack([c_x, c_y, s_k * fm_x, s_k * fm_y], dim=1))
            prior_boxes.append(torch.stack([c_x, c_y, s_k_prime * fm_x, s_k_prime * fm_y], dim=1))
            for a_r in self.aspect_ratios[i]:
                a_r_sqrt = math.sqrt(a_r)
                prior_boxes.append(torch.stack([c_x, c_y, s_k * fm_x * a_r_sqrt, s_k * fm_y / a_r_sqrt], dim=1))
                prior_boxes.append(torch.stack([c_x, c_y, s_k * fm_x / a_r_sqrt, s_k * fm_y * a_r_sqrt], dim=1))
        return prior_cox_crop(torch.concatenate(prior_boxes, dim=0), [self.w, self.h])
