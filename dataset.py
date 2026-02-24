import os
import torch
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from torchvision import tv_tensors

class ChestXRayClfDataset(Dataset):
    def __init__(self, data_root, covid_txt_path, healthy_txt_path, transform=True):
        self.filepaths = []
        self.labels = []
        self.transform = transform

        if os.path.exists(covid_txt_path):
            with open(covid_txt_path, 'r') as f:
                for line in f:
                    path = line.strip()
                    if os.path.exists(os.path.join(data_root, path)):
                        self.filepaths.append(os.path.join(data_root, path))
                        self.labels.append(1)
        
        if os.path.exists(healthy_txt_path):
            with open(healthy_txt_path, 'r') as f:
                for line in f:
                    path = line.strip()
                    if os.path.exists(os.path.join(data_root, path)):
                        self.filepaths.append(os.path.join(data_root, path))
                        self.labels.append(0)
        
        covid_count = sum(self.labels)
        healthy_count = len(self.labels) - covid_count
        print(f'Covid: {covid_count}, Healthy: {healthy_count}')
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index):
        img_path = self.filepaths[index]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            aug = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            img = aug(img)
        
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return img, label, img_path


class ChestXRaySegDataset(Dataset):
    def __init__(self, data_root, covid_txt_path, transform=None):
        self.filepaths = []
        self.transform = transform
        
        if os.path.exists(covid_txt_path):
            with open(covid_txt_path, 'r') as f:
                for line in f:
                    path = line.strip()
                    mask_path = path.replace("images", "annotations/lungVAE-masks")
                    mask_path = os.path.splitext(mask_path)[0] + "_mask.png"
                    if os.path.exists(os.path.join(data_root, path)) and os.path.exists(os.path.join(data_root, mask_path)):
                        self.filepaths.append(os.path.join(data_root, path))
        else:
            print(f"Warning: {covid_txt_path} not found.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        mask_path = img_path.replace("images", "annotations/lungVAE-masks")
        mask_path = os.path.splitext(mask_path)[0] + "_mask.png"

        # PIL natively throws a FileNotFoundError if these fail, so no need for `is None` checks
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Wrap them in v2 tv_tensors so the transforms know which one is the mask!
        img = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask)

        # Apply augmentations (v2 perfectly synchronizes the random flips for both)
        if self.transform:
            aug = v2.Compose([
                # Resizes both Image and Mask
                v2.Resize((256, 256), interpolation=v2.InterpolationMode.NEAREST),
                # Flips both synchronously
                v2.RandomHorizontalFlip(p=0.5),
                # Rotates both synchronously
                v2.RandomRotation(degrees=15, fill=0),
                # Converts both to tensors. Scales Image to [0,1], leaves Mask alone.
                v2.ToImage(), 
                v2.ToDtype(dtype=torch.float32, scale=True),
                # Normalizes ONLY the Image (it automatically ignores the Mask!)
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            img, mask = aug(img, mask)
        
        # Ensure mask is a Long tensor for CrossEntropyLoss/DiceLoss
        mask = (mask / 255).to(torch.long)
        
        return img, mask, img_path