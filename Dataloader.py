import os
import numpy as np
import glob
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time

import matplotlib.pyplot as plt
from IPython.display import clear_output



class DRIVE(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        'Initialization'
        self.transform = transform
        base_path = '/dtu/datasets1/02516/DRIVE/'
        data_path = os.path.join(base_path, 'training' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
        self.label_paths = sorted(glob.glob(data_path + '/1st_manual/*.gif'))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y

class PH2(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):

        self.data_path = os.path.join('/dtu/datasets1/02516/PH2_Dataset_images')   

        self.image_paths = sorted(glob.glob(os.path.join(self.data_path, '**', '*_Dermoscopic_Image/*.bmp'), recursive=True))
        self.lesion_paths = sorted(glob.glob(os.path.join(self.data_path, '**', '*_lesion/*.bmp'), recursive=True))
        self.roi_dirs = sorted(glob.glob(os.path.join(self.data_path, '**', '*_roi/*.bmp'), recursive=True))

        # Ensure all lists have the same length by filtering out mismatched entries
        min_length = min(len(self.image_paths), len(self.lesion_paths), len(self.roi_dirs))
        self.image_paths = self.image_paths[:min_length]
        self.lesion_paths = self.lesion_paths[:min_length]
        self.roi_dirs = self.roi_dirs[:min_length]
        self.transform = transform

        # Debug prints to check lengths
        # print(f'Length of image_paths: {len(self.image_paths)}')
        # print(f'Length of lesion_paths: {len(self.lesion_paths)}')
        # print(f'Length of roi_dirs: {len(self.roi_dirs)}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        lesion_path = self.lesion_paths[idx]

        # print(f'{idx = :}')
                    
        image = Image.open(image_path).convert("RGB")
        lesion = Image.open(lesion_path).convert("L")

        roi_path = self.roi_dirs[idx]  # Correctly index the roi_dirs list
        roi = Image.open(roi_path).convert("L")
        rois = [roi]

        if self.transform:
            image = self.transform(image)
            lesion = self.transform(lesion)
            # rois = [self.transform(r) for r in roi]
            rois = [self.transform(roi)]

        return image, lesion, rois



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

drive_dataset = DRIVE(train=True, transform=transform)
drive_loader = torch.utils.data.DataLoader(drive_dataset, batch_size=1, shuffle=True)

for index, (images, masks) in enumerate(drive_loader):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(images[0].permute(1, 2, 0))  # Convert from CHW to HWC
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(masks[0].squeeze(), cmap='gray')  
    plt.axis('off')

    # plt.show()
    plt.savefig(f'drive_sample_{str(index)}.png')
    # break  

# Test PH2 data loader similarly
ph2_dataset = PH2(train=True, transform=transform)
ph2_loader = torch.utils.data.DataLoader(ph2_dataset, batch_size=1, shuffle=True)


for index, (images, lesions, rois) in enumerate(DataLoader(ph2_dataset, batch_size=1, shuffle=True)):

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(images[0].permute(1, 2, 0))  # Convert from CHW to HWC
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Lesion Mask')
    plt.imshow(lesions[0].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('ROI Masks')
    for roi in rois[0]:
        plt.imshow(roi.squeeze(), cmap='gray', alpha=0.5)  # Overlay ROIs
    plt.axis('off')

    plt.savefig(f'ph2_sample_{str(index)}.png')
    # break  # Remove this to load more images
