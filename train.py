import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import os
import argparse

from models import DINOv3Classifier, DINOv3Segmenter 
from criterions import FocalDiceLoss, Metrics

from dataset import ChestXRayClfDataset, ChestXRaySegDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epoch=10, device='cpu', task='segmentation'):
    model = model.to(device)
    metric_calculator = Metrics()
    best_f1 = 0.0

    for epoch in range(num_epoch):
        ### Training ###
        model.train() 
        if hasattr(model, 'backbone'):
            model.backbone.eval() # Keep foundation features frozen

        train_loss = 0.0
        train_metrics_sum = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        train_batches = len(train_loader)

        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch} [Train]"):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_metrics = metric_calculator(outputs, targets)
            for k in train_metrics_sum:
                train_metrics_sum[k] += batch_metrics[k]

        # Calculate epoch averages
        epoch_train_loss = train_loss / train_batches
        epoch_train_f1 = train_metrics_sum['f1'] / train_batches
        epoch_train_acc = train_metrics_sum['accuracy'] / train_batches
        print(f"\n[Train] Loss: {epoch_train_loss:.4f} | F1: {epoch_train_f1:.4f} | Acc: {epoch_train_acc:.4f}")

        ### Validation ###
        model.eval()
        val_loss = 0.0
        val_metrics_sum = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        val_batches = len(val_loader)

        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epoch} [Val]"):
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)

                loss = loss_func(outputs, targets)
                val_loss += loss.item()
                
                # Accumulate metrics
                batch_metrics = metric_calculator(outputs, targets)
                for k in val_metrics_sum:
                    val_metrics_sum[k] += batch_metrics[k]

        # Calculate epoch averages
        epoch_val_loss = val_loss / val_batches
        epoch_val_f1 = val_metrics_sum['f1'] / val_batches
        epoch_val_acc = val_metrics_sum['accuracy'] / val_batches
        
        print(f"[Val] Loss: {epoch_val_loss:.4f} | F1: {epoch_val_f1:.4f} | Acc: {epoch_val_acc:.4f}\n")

        # Save best model based on F1 Score
        if epoch_val_f1 > best_f1:
            best_f1 = epoch_val_f1
            saved_folder = "./saved_models"
            os.makedirs(saved_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(saved_folder, f"best_model_{task}.pth"))
            print(f"New best model saved with F1: {best_f1:.4f}")

    print(f"Training Complete. Best Validation F1: {best_f1:.4f}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training on ChestXRay Dataset with DINOv3")
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation'], default='segmentation', help='Task type: classification or segmentation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--data_root', type=str, default='/path/to/data_root', help='Root directory of the dataset')
    parser.add_argument('--model_name', type=str, default='dinov3_vitl16', help='DINOv3 model variant to use')
    parser.add_argument('--weight_focal', type=float, default=1.0, help='Weight for Focal Loss component')
    parser.add_argument('--weight_dice', type=float, default=1.0, help='Weight for Dice Loss component')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer')


    args = parser.parse_args()

    DATA_ROOT = args.data_root

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if args.task == 'classification':
        model = DINOv3Classifier(model_name=args.model_name, num_classes=2)
        dataset = ChestXRayClfDataset(
            data_root=DATA_ROOT,
            covid_txt_path=os.path.join(DATA_ROOT, 'COVID.txt'), 
            healthy_txt_path=os.path.join(DATA_ROOT, 'healthy.txt'),
            transform=True
        )
        loss_func = torch.nn.CrossEntropyLoss()
    elif args.task == 'segmentation':
        model = DINOv3Segmenter(model_name=args.model_name, num_classes=2)
        dataset = ChestXRaySegDataset(
            data_root=DATA_ROOT,
            covid_txt_path=os.path.join(DATA_ROOT, 'COVID.txt'), 
            transform=True
        )
        loss_func = FocalDiceLoss(weight_focal=args.weight_focal, weight_dice=args.weight_dice)
    else:
        raise ValueError("Invalid task type. Choose 'classification' or 'segmentation'.")
    
    
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    

    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    if len(train_loader) > 0 and len(val_loader) > 0:
        train_model(model, train_loader, val_loader, loss_func, optimizer, device=device, num_epoch=args.epochs, task=args.task)
    else:
        print("Loaders not defined. Please pass actual DataLoaders to train_model.")