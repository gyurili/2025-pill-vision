import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(num_classes):
    """
    Pretrained Faster R-CNN 모델을 불러온 뒤, 출력층을 수정하는 함수

    Args:
        num_classes (int): 예측할 클래스 개수 (배경 포함)

    Returns:
        model (torch.nn.Module): 수정된 Faster R-CNN 모델
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
