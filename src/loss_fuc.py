import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment


def generalized_box_iou(boxes1, boxes2):
    min_box = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    max_box = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    inter = (torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) -
             torch.max(boxes1[:, None, :2], boxes2[:, :2])).clamp(min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter_area
    iou = inter_area / (union + 1e-6)

    convex_area = ((max_box[:, :, 0] - min_box[:, :, 0]) *
                   (max_box[:, :, 1] - min_box[:, :, 1]))
    convex_area = torch.clamp(convex_area, min=1e-6)  # 안정화

    giou = iou - (convex_area - union) / convex_area
    giou = torch.clamp(giou, min=-1.0, max=1.0)  # 안정화

    return giou


class HungarianMatcher:
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2, img_size=640):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.img_size = img_size

    def __call__(self, outputs, targets):
        return self.forward(outputs, targets)

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size = len(outputs["pred_logits"])
        indices = []

        for i in range(batch_size):
            pred_logits = outputs["pred_logits"][i].softmax(-1)
            pred_boxes = outputs["pred_boxes"][i].clone()
            target_labels = targets[i]["labels"]
            target_boxes = targets[i]["boxes"]

            # 정규화 해제 (0~1 → 픽셀)
            pred_boxes[:, [0, 2]] *= self.img_size
            pred_boxes[:, [1, 3]] *= self.img_size

            cost_class = -pred_logits[:, target_labels]
            cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
            cost_giou = -generalized_box_iou(pred_boxes, target_boxes)

            cost_matrix = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )

            # NaN 또는 Inf 발생 시 디버깅 정보 출력
            if torch.isnan(cost_matrix).any() or torch.isinf(cost_matrix).any():
                print("==== [NaN or Inf detected in cost_matrix] ====")
                print("pred_logits:\n", pred_logits)
                print("pred_boxes:\n", pred_boxes)
                print("target_labels:\n", target_labels)
                print("target_boxes:\n", target_boxes)
                print("cost_class:\n", cost_class)
                print("cost_bbox:\n", cost_bbox)
                print("cost_giou:\n", cost_giou)
                print("cost_matrix:\n", cost_matrix)

                cost_matrix[torch.isnan(cost_matrix)] = 0
                cost_matrix[torch.isinf(cost_matrix)] = 0

            row_idx, col_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.int64),
                torch.as_tensor(col_idx, dtype=torch.int64)
            ))

        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.img_size = img_size

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.cat([
            t["labels"][J] for t, (_, J) in zip(targets, indices)
        ])
        loss_ce = F.cross_entropy(src_logits[idx], target_classes, reduction="mean")
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)

        # 예측 박스 정규화 해제
        src_boxes = outputs["pred_boxes"][idx].clone()
        src_boxes[:, [0, 2]] *= self.img_size
        src_boxes[:, [1, 3]] *= self.img_size

        target_boxes = torch.cat([
            t["boxes"][J] for t, (_, J) in zip(targets, indices)
        ], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="mean")

        giou = torch.diag(generalized_box_iou(src_boxes, target_boxes))
        valid_mask = torch.isfinite(giou)
        if valid_mask.any():
            loss_giou = 1 - giou[valid_mask].mean()
        else:
            loss_giou = torch.tensor(0.0, device=src_boxes.device)

        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices))
        return losses


def get_loss(num_classes=82, img_size=640):
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, img_size=img_size)
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_bbox": 5,
            "loss_giou": 2
        },
        eos_coef=0.1,
        losses=["labels", "boxes"],
        img_size=img_size
    )
    return criterion
