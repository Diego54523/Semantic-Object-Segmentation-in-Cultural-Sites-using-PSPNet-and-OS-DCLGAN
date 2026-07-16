import os
import numpy as np
from PIL import Image
from dataset.dataset import CulturalDataset
from collections import Counter

def check_raw_images(labels_dir, num_samples=10):
    print(f"\n--- Checking Raw Files in: {labels_dir} ---")
    if not os.path.exists(labels_dir):
        print("FOLDER NOT FOUND!")
        return
    
    files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
    if not files:
        print("No masks found.")
        return

    modes = Counter()
    for f in files[:num_samples]:
        img = Image.open(os.path.join(labels_dir, f))
        modes[img.mode] += 1
    
    for mode, count in modes.items():
        print(f"Found {count} images with PIL.mode = '{mode}'")
        if mode == 'RGB':
            print("  WARNING: Masks are RGB. Loader will map via colormap.")
        elif mode in ['P', 'L', 'I']:
            print("  OK: Masks are indexed or grayscale.")

def check_dataset_loader(dataset, domain_name, num_samples=100):
    print(f"\n==================================================")
    print(f" DATASET ANALYSIS: {domain_name.upper()} ")
    print(f"==================================================")
    
    class_hist = np.zeros(26) # Classes 0-24 + Index 25 (for 255)
    total_pixels = 0
    missing_labels_count = 0
    
    samples_to_check = min(len(dataset), num_samples)
    print(f"Analyzing {samples_to_check} samples...")

    for i in range(samples_to_check):
        _, label_tensor = dataset[i]
        
        # Map 255 to 25 for histogram counting
        label_np = label_tensor.numpy().copy()
        label_np[label_np == 255] = 25 
        
        unique, counts = np.unique(label_np, return_counts=True)
        
        # Check for empty masks (only contains index 25/255)
        if len(unique) == 1 and unique[0] == 25:
            missing_labels_count += 1
            
        for u, c in zip(unique, counts):
            if u <= 25:
                class_hist[u] += c
                total_pixels += c
                
    print("\n--- CLASS DISTRIBUTION IN DATALOADER ---")
    for i in range(25):
        if class_hist[i] > 0:
            percentage = (class_hist[i] / total_pixels) * 100
            print(f"  Class {i:2d}: {class_hist[i]:10.0f} pixels ({percentage:.2f}%)")
            
    if class_hist[25] > 0:
        percentage = (class_hist[25] / total_pixels) * 100
        print(f"\n  [IGNORE] 255: {class_hist[25]:10.0f} pixels ({percentage:.2f}%)")
        
    print(f"\n--- MASK STATISTICS ---")
    print(f"Completely EMPTY masks (only 255): {missing_labels_count} out of {samples_to_check}")

if __name__ == '__main__':
    dataset_path = '../EGO-CH-OBJ-SEG/EGO-CH-OBJ-SEG'
    
    synth_labels_dir = os.path.join(dataset_path, 'synthetic', 'train', 'labels')
    real_labels_dir = os.path.join(dataset_path, 'real', 'train', 'labels')
    
    check_raw_images(synth_labels_dir)
    check_raw_images(real_labels_dir)
    
    print("\nLoading datasets...")
    try:
        dataset_synth = CulturalDataset(dataset_path, domain='synthetic', split='train')
        check_dataset_loader(dataset_synth, "Synthetic", num_samples=100)
    except Exception as e:
        print(f"Error loading Synthetic dataset: {e}")

    try:
        dataset_real = CulturalDataset(dataset_path, domain='real', split='train')
        check_dataset_loader(dataset_real, "Real", num_samples=100)
    except Exception as e:
        print(f"Error loading Real dataset: {e}")
        
    print("\n--- DEBUG COMPLETED ---")