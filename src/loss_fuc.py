import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU 계산 함수.
    Args:
        boxes1 (Tensor): 첫 번째 바운딩 박스 집합 (N, 4).
        boxes2 (Tensor): 두 번째 바운딩 박스 집합 (M, 4).
    Returns:
        Tensor: Generalized IoU 값 (N, M).
    """
    # 최소 좌표 및 최대 좌표 계산
    min_box = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    max_box = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 교차 영역 계산
    inter = (torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])).clamp(0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    # 각 박스 면적 계산
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # IoU 및 Generalized IoU 계산
    union = area1[:, None] + area2 - inter_area
    iou = inter_area / (union + 1e-6)  # NaN 방지
    convex_area = (max_box[:, :, 0] - min_box[:, :, 0]) * (max_box[:, :, 1] - min_box[:, :, 1])

    return iou - (convex_area - union) / (convex_area + 1e-6)  # NaN 방지


class HungarianMatcher:
    """
    헝가리안 알고리즘을 이용한 객체 매칭.
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): 모델 출력.
            targets (list): 실제 객체 정보.
        Returns:
            list: 각 배치의 매칭 인덱스 리스트.
        """
        batch_size = len(outputs["pred_logits"])
        indices = []

        for i in range(batch_size):
            pred_logits = outputs["pred_logits"][i].softmax(-1)
            pred_boxes = outputs["pred_boxes"][i]
            target_labels = targets[i]["labels"]
            target_boxes = targets[i]["boxes"]

            cost_class = -pred_logits[:, target_labels]
            cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
            cost_giou = -generalized_box_iou(pred_boxes, target_boxes)

            # cost_matrix 결합
            cost_matrix = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            # NaN 또는 Inf 값 확인 및 처리
            if torch.isnan(cost_matrix).any() or torch.isinf(cost_matrix).any():
                print("Warning: cost_matrix contains NaN or Inf values! Replacing with zeros.")
                cost_matrix[torch.isnan(cost_matrix)] = 0
                cost_matrix[torch.isinf(cost_matrix)] = 0

            row_idx, col_idx = linear_sum_assignment(cost_matrix.cpu().numpy())

            indices.append((torch.as_tensor(row_idx, dtype=torch.int64),
                            torch.as_tensor(col_idx, dtype=torch.int64)))

        return indices


class SetCriterion(torch.nn.Module):
    """
    DETR 학습을 위한 손실 함수.
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

    def loss_labels(self, outputs, targets, indices):
        """
        CrossEntropy를 이용한 Classification Loss.
        """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        loss_ce = F.cross_entropy(src_logits[idx], target_classes, reduction="mean")
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        """
        Bounding Box Loss (L1 + GIoU).
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="mean")
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes)).mean()

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        """
        Hungarian Matcher를 통해 매칭 후 손실 계산.
        """
        indices = self.matcher.forward(outputs, targets)
        losses = {}

        for loss in self.losses:
            losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices))

        return losses

    def _get_src_permutation_idx(self, indices):
        """
        indices로부터 배치 차원의 인덱스 변환.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def get_loss(num_classes=82):
    """
    Hungarian Matcher와 SetCriterion을 생성하여 반환.
    
    Args:
        num_classes (int): 총 클래스 개수.

    Returns:
        SetCriterion: 학습 손실 함수.
    """
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
        eos_coef=0.1,
        losses=["labels", "boxes"]
    )

    return criterion