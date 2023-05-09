import torch
from torch import Tensor

def box_area(boxes: Tensor) -> Tensor:
    return torch.multiply(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])

def compute_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    '''
    输入GT框boxes1(size = [N, 4])和待匹配的先验框boxes2(size = [M, 4]) 形式为(xmin, ymin, xmax, ymax)
    返回一个size = [N, M]的iou矩阵
    '''
    box_area1 = box_area(boxes1)
    box_area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = box_area1[:, None] + box_area2 - inter

    iou = inter / union
    return iou

class SSDMatcher:
    def __init__(
        self,
        num_classes: int = 21,
        nms_threshold: float = 0.5,
        top_k: int = 400
    ):
        self.num_classes = num_classes
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.BELOW_IOU_THRESHOLD = -1
        self.GE_IOU_THRESHOLD = -2 # GE = greater than

    def __call__(
        self,
        iou_matrix: Tensor
    ):
        print(iou_matrix)
        # iou_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        best_predicted_values, best_predicted_positions = iou_matrix.max(dim=0) # shape:(N)
        print(best_predicted_values, best_predicted_positions)

        below_nms_threshold = best_predicted_values < self.nms_threshold
        ge_nms_threshold = best_predicted_values >= self.nms_threshold
        print(below_nms_threshold, ge_nms_threshold)
        best_predicted_positions[below_nms_threshold] = self.BELOW_IOU_THRESHOLD
        best_predicted_positions[ge_nms_threshold] = self.GE_IOU_THRESHOLD
        print(best_predicted_positions)

        # Max over predicted elements (dim 0) to find best gt candidate for each prediction
        best_gt_values, best_gt_positions = iou_matrix.max(dim=1) # shape:(M)
        print(best_gt_positions)

        best_predicted_positions[best_gt_positions] = torch.arange(
            best_gt_positions.size(0), dtype=torch.int64, device=best_gt_positions.device
        )

        return best_predicted_positions

mat = Tensor(
    [[0.2,0.7,0.3,0.4],[0.44,0.2,0.5,0.3],[0.1,0.2,0.7,0.2]]
)

a = SSDMatcher()
b = a(mat)
print(b)
print(torch.where(b >= 0)[0])