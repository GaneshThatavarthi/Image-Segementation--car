import os
import cv2
import numpy as np
import torch

class SegmentationDataset:
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]

        # Paths
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file.replace(".jpg", ".png"))

        # Load image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.size, self.size))
        img = img / 255.0   # normalize

        # Load mask (grayscale)
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.size, self.size))

        # Convert to tensor
        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        return img, mask

if __name__ == "__main__":
    # Initialize the dataset
    dataset = SegmentationDataset(img_dir="images/train", mask_dir="masks/train", size=256)
    
    # Grab the very first image and mask from our dataset generator
    img, mask = dataset[0]

    # Print the shapes as you requested!
    print("Dataset successfully initialized!")
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)