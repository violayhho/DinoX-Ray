import torch
import torch.nn as nn
import torch.nn.functional as F

import os


def get_dinov3_backbone(model_name='dinov3_vitl16'):
    """
    Loads a DINOv3 backbone from PyTorch Hub.
    Options include 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16', 
    and ConvNeXt variants like 'dinov3_convnext_tiny'.
    """
    if model_name not in ['dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16', 
                          'dinov3_vitl16','dinov3_vith16plus', 'dinov3_vit7b16']:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    models_folder = '/path/to/dinov3/'
    pretrained_weights_folder = os.path.join(models_folder, 'dinov3/pretrained_weight')

    if model_name == 'dinov3_vits16':
        weights = os.path.join(pretrained_weights_folder, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
    if model_name == 'dinov3_vits16plus':
        weights = os.path.join(pretrained_weights_folder, 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
    if model_name == 'dinov3_vitb16':
        weights = os.path.join(pretrained_weights_folder, 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    if model_name == 'dinov3_vitl16':
        weights = os.path.join(pretrained_weights_folder, 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    if model_name == 'dinov3_vith16plus':
        weights = os.path.join(pretrained_weights_folder, 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    if model_name == 'dinov3_vit7b16':
        weights = os.path.join(pretrained_weights_folder, 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth')

    backbone = torch.hub.load(models_folder, model_name, source="local", weights=weights)
    
    for param in backbone.parameters():
        param.requires_grad = False
        
    return backbone


class DINOv3Classifier(nn.Module):
    def __init__(self, model_name='dinov3_vitl16', num_classes=2):
        super(DINOv3Classifier, self).__init__()
        self.backbone = get_dinov3_backbone(model_name)

        if 'vits' in model_name:
            embed_dim = 384
        elif 'vitb' in model_name:
            embed_dim = 768
        elif 'vitl' in model_name:
            embed_dim = 1024
        elif 'vith' in model_name:
            embed_dim = 1280
        elif 'vit7b' in model_name:
            embed_dim = 1792
        
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        
        # Use the global classification token ([CLS] token)
        cls_token = features['x_norm_clstoken']
        
        logits = self.head(cls_token)
        return logits


class DINOv3Segmenter(nn.Module):
    def __init__(self, model_name='dinov3_vitl16', num_classes=2, patch_size=16):
        super(DINOv3Segmenter, self).__init__()
        self.backbone = get_dinov3_backbone(model_name)
        self.patch_size = patch_size
        
        if 'vits' in model_name:
            embed_dim = 384
        elif 'vitb' in model_name:
            embed_dim = 768
        elif 'vitl' in model_name:
            embed_dim = 1024
        elif 'vith' in model_name:
            embed_dim = 1280
        elif 'vit7b' in model_name:
            embed_dim = 4096
        
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        features = self.backbone.forward_features(x)
        
        # Extract the dense spatial patch tokens (ignoring CLS and register tokens)
        patch_tokens = features['x_norm_patchtokens']
        
        # Reshape the flat patch tokens back to a 2D spatial grid
        H_feat = H // self.patch_size
        W_feat = W // self.patch_size
        
        # [Batch, Num_Patches, Embed_Dim] -> [Batch, Embed_Dim, H_feat, W_feat]
        patch_grid = patch_tokens.permute(0, 2, 1).reshape(B, -1, H_feat, W_feat)
        
        logits = self.decoder(patch_grid)
        
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits