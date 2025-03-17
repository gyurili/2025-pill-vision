import torch.nn as nn
from torchvision.models.resnet import resnet50
from torch.nn import TransformerEncoderLayer
from torchvision.ops import DeformConv2d

class DeformableAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.conv_offset = nn.Conv2d(hidden_dim, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        offset = self.conv_offset(x)
        x = self.deform_conv(x, offset)
        x = x.flatten(2).permute(2, 0, 1)  # Reshape for transformer
        x, _ = self.attn(x, x, x)
        return x

class Backbone(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        return x

class DeformableDETR(nn.Module):
    def __init__(self, num_classes=82, hidden_dim=256, num_queries=100, num_heads=8, num_layers=6):
        super().__init__()
        self.backbone = Backbone(hidden_dim)
        self.deform_attn = DeformableAttention(hidden_dim, num_heads)
        
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.fc_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.fc_bbox = nn.Linear(hidden_dim, 4)  # [x, y, w, h]
        
    def forward(self, images):
        x = self.backbone(images)
        x = self.deform_attn(x)
        
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, images.shape[0], 1)
        hs = self.transformer(queries + x)
        
        logits = self.fc_class(hs)
        bboxes = self.fc_bbox(hs).sigmoid()
        
        return logits, bboxes