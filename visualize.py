import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse

from models import DINOv3Classifier, DINOv3Segmenter
from dataset import ChestXRayClfDataset, ChestXRaySegDataset

def unnormalize(tensor):
    """
    Reverses the ImageNet normalization applied in dataset.py .
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Unnormalize and move channels to the end (C, H, W) -> (H, W, C)
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()

def load_validation_data(data_root, dataset_class, txt_paths, batch_size=4):
    """
    Recreates the exact validation split used during training by 
    fixing the random_state.
    """
    dataset = dataset_class(data_root=data_root, **txt_paths, transform=True)
    
    _, val_indices = train_test_split(
        list(range(len(dataset))), 
        test_size=0.2, 
        random_state=42
    )
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    return DataLoader(val_subset, batch_size=batch_size, shuffle=True)

def visualize_classification(model, dataloader, device, output_dir):
    print("Running Classification Inference...")
    model.eval()
    
    images, targets = None, None
    for batch_images, batch_targets, _ in dataloader:
        # Check if this batch has AT LEAST one COVID image (label 1)
        if 1 in batch_targets:
            images = batch_images
            targets = batch_targets
            # If it ALSO has a Healthy image (label 0), it's a perfect mixed batch, so stop searching
            if 0 in batch_targets:
                break
                
    if images is None:
        images, targets, _ = next(iter(dataloader))
    
    images, targets = images.to(device), targets.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle("Classification Results (Validation Set)", fontsize=16, fontweight='bold')
    
    classes = ['Healthy', 'COVID']
    
    for i in range(4):
        img_np = unnormalize(images[i])
        true_label = classes[targets[i].item()]
        pred_label = classes[preds[i].item()]
        
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        axes[i].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.90]) 
    plt.savefig(os.path.join(output_dir, "demo_classification.png"), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_segmentation(model, dataloader, device, output_dir):
    print("Running Segmentation Inference...")
    model.eval()
    
    images, masks, _ = next(iter(dataloader))
    images = images.to(device)
    masks = masks.squeeze(1).cpu().numpy() # [B, H, W]
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
    fig, axes = plt.subplots(4, 3, figsize=(9, 12)) 
    fig.suptitle("Segmentation Results (Validation Set)", fontsize=16, fontweight='bold')
    
    col_titles = ["Input Image", "Ground Truth Mask", "Predicted Mask"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, pad=10)
        
    for i in range(4):
        img_np = unnormalize(images[i])
        
        # Original Image
        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(preds[i], cmap='gray')
        axes[i, 2].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=0.5, h_pad=1.0) 
    plt.savefig(os.path.join(output_dir, "demo_segmentation.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization of DINOv3 Classification and Segmentation on ChestXRay Dataset")
    parser.add_argument('--data_root', type=str, default='/path/to/data_root', help='Root directory of the dataset')
    args = parser.parse_args()
    DATA_ROOT = args.data_root
    CLF_WEIGHTS = "./saved_models/best_model_classification.pth" 
    SEG_WEIGHTS = "./saved_models/best_model_segmentation.pth"
    OUTPUT_DIR = "./visualization_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Classification Demo setup
    if os.path.exists(CLF_WEIGHTS):
        clf_model = DINOv3Classifier(num_classes=2).to(device)
        clf_model.load_state_dict(torch.load(CLF_WEIGHTS, map_location=device))
        
        clf_loader = load_validation_data(
            DATA_ROOT, 
            ChestXRayClfDataset, 
            {'covid_txt_path': os.path.join(DATA_ROOT, 'COVID.txt'), 
             'healthy_txt_path': os.path.join(DATA_ROOT, 'healthy.txt')}
        )
        visualize_classification(clf_model, clf_loader, device, OUTPUT_DIR)
    else:
        print(f"Skipping Classification Demo: Could not find {CLF_WEIGHTS}")

    # 2. Segmentation Demo setup
    if os.path.exists(SEG_WEIGHTS):
        seg_model = DINOv3Segmenter(num_classes=2).to(device)
        seg_model.load_state_dict(torch.load(SEG_WEIGHTS, map_location=device))
        
        seg_loader = load_validation_data(
            DATA_ROOT, 
            ChestXRaySegDataset, 
            {'covid_txt_path': os.path.join(DATA_ROOT, 'COVID.txt')}
        )
        visualize_segmentation(seg_model, seg_loader, device, OUTPUT_DIR)
    else:
        print(f"Skipping Segmentation Demo: Could not find {SEG_WEIGHTS}")