import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# LOSS FUNCTIONS
# ==========================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        
        # Binary Case
        if num_classes == 1:
            inputs_soft = torch.sigmoid(inputs).view(-1)
            targets_flat = targets.view(-1).float()
            
            intersection = (inputs_soft * targets_flat).sum()
            cardinality = inputs_soft.sum() + targets_flat.sum()
        
        # Multi-class Case
        else:
            inputs_soft = F.softmax(inputs, dim=1)
            targets = targets.long()
            
            # Ensure target is [B, H, W] for one_hot
            if targets.dim() == inputs.dim():
                targets = targets.squeeze(1) 
                
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            # Sum over Batch and Spatial Dimensions, leaving Class dimension intact
            dims = (0, 2, 3) if inputs.dim() == 4 else (0,)
            
            intersection = torch.sum(inputs_soft * targets_one_hot, dim=dims)
            cardinality = torch.sum(inputs_soft + targets_one_hot, dim=dims)
            
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        
        # Binary Case
        if num_classes == 1:
            inputs = inputs.view(-1)
            targets = targets.view(-1).float()
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Multi-class Case
        else:
            targets = targets.long()
            if targets.dim() == inputs.dim():
                targets = targets.squeeze(1)
            loss = F.cross_entropy(inputs, targets, reduction='none')
            
        pt = torch.exp(-loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class FocalDiceLoss(nn.Module):
    def __init__(self, weight_focal=1.0, weight_dice=1.0, alpha=0.25, gamma=2.0, smooth=1e-5):
        super(FocalDiceLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return (self.weight_focal * focal_loss) + (self.weight_dice * dice_loss)

# ==========================================
# EVALUATION METRICS
# ==========================================

class Metrics:
    """
    Computes Accuracy, Precision, Recall, and F1 Score.
    Automatically handles both Binary and Multi-class Macro-averaging.
    """
    def __init__(self, threshold=0.5, epsilon=1e-7):
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, logits, targets):
        num_classes = logits.shape[1]
        
        # Binary Case
        if num_classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float().view(-1)
            targets = targets.float().view(-1)
            
            tp = (preds * targets).sum().item()
            fp = (preds * (1 - targets)).sum().item()
            fn = ((1 - preds) * targets).sum().item()
            tn = ((1 - preds) * (1 - targets)).sum().item()
            
            accuracy = (tp + tn) / (tp + fp + fn + tn + self.epsilon)
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
            
        # Multi-class Case (Macro Average)
        else:
            preds = torch.argmax(logits, dim=1).view(-1)
            targets = targets.long()
            if targets.dim() == logits.dim():
                targets = targets.squeeze(1)
            targets = targets.view(-1)
            
            accuracy = (preds == targets).float().mean().item()
            
            precision_sum, recall_sum, f1_sum = 0.0, 0.0, 0.0
            
            for c in range(num_classes):
                p_c = (preds == c).float()
                t_c = (targets == c).float()
                
                tp = (p_c * t_c).sum().item()
                fp = (p_c * (1 - t_c)).sum().item()
                fn = ((1 - p_c) * t_c).sum().item()
                
                prec_c = tp / (tp + fp + self.epsilon)
                rec_c = tp / (tp + fn + self.epsilon)
                f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c + self.epsilon)
                
                precision_sum += prec_c
                recall_sum += rec_c
                f1_sum += f1_c
                
            precision = precision_sum / num_classes
            recall = recall_sum / num_classes
            f1 = f1_sum / num_classes

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }