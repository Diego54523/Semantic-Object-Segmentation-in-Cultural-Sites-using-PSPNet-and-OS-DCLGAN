import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

def decode_segmentation_mask(label_pil):
    # Use directly if grayscale or indexed
    if label_pil.mode in ['L', 'P', 'I']:
        return np.array(label_pil, dtype=np.int64)
        
    # Convert to RGB and int32 to prevent overflow
    label_rgb = np.array(label_pil.convert("RGB"), dtype=np.int32)
    
    # Handle grayscale saved as RGB
    if np.array_equal(label_rgb[:, :, 0], label_rgb[:, :, 1]) and \
       np.array_equal(label_rgb[:, :, 1], label_rgb[:, :, 2]):
        return label_rgb[:, :, 0].astype(np.int64)
        
    # EGO-CH-OBJ-SEG exact palette
    colormap = np.array([
        [  0,   0,   0],  # Class 0 (Background)
        [  0, 117,  21],  # Class 1
        [  0, 219, 154],  # Class 2
        [  7, 255,   0],  # Class 3
        [ 17,  48,  40],  # Class 4
        [ 18,  10,  17],  # Class 5
        [ 27,  43,   0],  # Class 6
        [ 33,   0,  38],  # Class 7
        [ 40,  43,  36],  # Class 8
        [ 42,  79,   0],  # Class 9
        [ 54,  48,  15],  # Class 10
        [ 59,  41,  53],  # Class 11
        [ 78, 242, 255],  # Class 12
        [100, 107,  88],  # Class 13
        [117,  72,  56],  # Class 14
        [135,   0, 117],  # Class 15
        [140, 110, 118],  # Class 16
        [161,  86, 118],  # Class 17
        [183, 184, 182],  # Class 18
        [202, 240,  14],  # Class 19
        [213, 195, 242],  # Class 20
        [219, 168, 111],  # Class 21
        [240,  37,  98],  # Class 22
        [242,  89,  48],  # Class 23
        [255,   0,   0],  # Class 24
    ], dtype=np.int32)
    
    # Compute squared Euclidean distance
    diff = label_rgb[:, :, np.newaxis, :] - colormap[np.newaxis, np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=-1) 
    
    # Find nearest color index
    label_idx = np.argmin(distances, axis=-1)
    
    # Ignore pixels far from any known color (threshold > 1500)
    min_distances = np.min(distances, axis=-1)
    label_idx[min_distances > 1500] = 255
    
    return label_idx.astype(np.int64)

class CulturalDataset(Dataset):
    def __init__(self, root_dir, domain='synthetic', split='train', image_size=256, percentage=1.0):
        self.split = split
        self.frames_dir = os.path.join(root_dir, domain, split, 'frames')
        self.labels_dir = os.path.join(root_dir, domain, split, 'labels')
        self.image_filenames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')])
        
        # Percentage filtering logic
        if split == 'train' and percentage < 1.0:
            num_samples = max(1, int(len(self.image_filenames) * percentage))
            random.seed(42)  # Use seed for reproducible sampling
            self.image_filenames = random.sample(self.image_filenames, num_samples)
            random.seed() 
            print(f"[{domain.upper()}] Sampled {percentage*100}%: {num_samples} images.")

        self.image_size = image_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.frames_dir, img_name)
        label_name = img_name.replace('.jpg', '.png')
        label_path = os.path.join(self.labels_dir, label_name)

        image = Image.open(img_path).convert("RGB")
        
        if os.path.exists(label_path):
            label = Image.open(label_path)
        else:
            label = Image.new("L", image.size, color=255) # Empty mask to ignore

        # Data Augmentation
        if self.split == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            image = jitter(image)

        image = TF.resize(image, (self.image_size, self.image_size))
        label = TF.resize(label, (self.image_size, self.image_size), interpolation=T.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        label_np = decode_segmentation_mask(label)
        label_tensor = torch.as_tensor(label_np, dtype=torch.long)

        return image, label_tensor