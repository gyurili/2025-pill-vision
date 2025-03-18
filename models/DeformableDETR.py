import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from src import device


class DeformableAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.deform_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        with autocast(device.type):
            x = self.deform_conv(x)
            x = x.flatten(2).permute(2, 0, 1)
            x, _ = self.attn(x, x, x)
        return x

class Backbone(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

class DeformableDETR(nn.Module):
    def __init__(self, num_classes=82, hidden_dim=256, num_queries=5, num_heads=8, num_layers=6):
        super().__init__()
        self.backbone = Backbone(hidden_dim)
        self.deform_attn = DeformableAttention(hidden_dim, num_heads)

        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.fc_class = nn.Linear(hidden_dim, num_classes + 1)
        self.fc_bbox = nn.Linear(hidden_dim, 4)

    def forward(self, images):
        with autocast(device.type):
            x = self.backbone(images)
            x = self.deform_attn(x).float()

        memory = self.transformer_encoder(x)

        queries = self.query_embed.weight.unsqueeze(1).repeat(1, images.shape[0], 1)
        hs = self.transformer_decoder(queries, memory)

        logits = self.fc_class(hs).permute(1, 0, 2)
        bboxes = self.fc_bbox(hs).sigmoid().permute(1, 0, 2)

        return {
            "pred_logits": logits,
            "pred_boxes": bboxes
        }