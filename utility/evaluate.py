import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import CulturalDataset
from models.PSPNet import PSPNet

load_dotenv()

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- INFERENCE AND EVALUATION ON DEVICE: {device} ---")

    NUM_CLASSES = 25
    BATCH_SIZE = 8 
    NUM_WORKERS = 8
    dataset_path = os.getenv('DATA_PATH', './EGO-CH-OBJ-SEG/EGO-CH-OBJ-SEG')
    
    print("Loading real validation dataset...")
    dataset_real_val = CulturalDataset(dataset_path, domain='real', split='val')
    if len(dataset_real_val) == 0:
        print("Folder 'val' empty, trying 'test'...")
        dataset_real_val = CulturalDataset(dataset_path, domain='real', split='test')

    if len(dataset_real_val) == 0:
        print("WARNING: No validation images found.")
        return

    loader_real_val = DataLoader(dataset_real_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Found {len(dataset_real_val)} validation images.")
    
    models_to_evaluate = {
        "0% Real Data": "weights/pspnet_final.pth", 
        "50% Real Data": "weights/pspnet_final_real50.pth", 
        "100% Real Data": "weights/pspnet_final_real100.pth"
    }

    psp_net = PSPNet(layers=50, classes=NUM_CLASSES).to(device)

    for model_name, weight_path in models_to_evaluate.items():
        print(f"\n{'='*70}")
        print(f" EVALUATING MODEL: {model_name}")
        print(f"{'='*70}")

        if not os.path.exists(weight_path):
            print(f" ERROR: Weights file not found: {weight_path}")
            continue

        print(f"Loading weights from: {weight_path}...")
        try:
            state_dict = torch.load(weight_path, map_location=device)
            new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
            psp_net.load_state_dict(new_state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            continue

        psp_net.eval()
        all_preds, all_targets = [], []

        print("Starting inference...")
        with torch.no_grad():
            for batch_idx, (img_real, label_real) in enumerate(loader_real_val):
                img_real, label_real = img_real.to(device), label_real.to(device)
                label_real[(label_real >= NUM_CLASSES) & (label_real != 255)] = 255

                # Forward pass
                img_real_psp = F.interpolate(img_real, size=(473, 473), mode='bilinear', align_corners=True)
                outputs_psp = psp_net(img_real_psp)
                outputs = F.interpolate(outputs_psp, size=label_real.shape[1:], mode='bilinear', align_corners=True)
                
                pred_classes = torch.argmax(outputs, dim=1)
                
                # Flatten and filter ignored pixels
                pred_flat = pred_classes.cpu().numpy().flatten()
                label_flat = label_real.cpu().numpy().flatten()
                
                mask = (label_flat != 255)
                all_preds.extend(pred_flat[mask])
                all_targets.extend(label_flat[mask])

                if (batch_idx + 1) % 10 == 0:
                    print(f" - Processed batch {batch_idx + 1}/{len(loader_real_val)}")

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        unique_classes = np.arange(NUM_CLASSES)
        class_names = [f"Class_{c}" for c in unique_classes]
        
        cm = confusion_matrix(all_targets, all_preds, labels=unique_classes)
        ious, accuracies = [], []
        
        for i, cls in enumerate(unique_classes):
            intersection = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - intersection
            ground_truth = cm[i, :].sum()
            
            iou = float(intersection) / float(union) if union > 0 else 0.0
            acc = float(intersection) / float(ground_truth) if ground_truth > 0 else 0.0
            
            ious.append(iou)
            accuracies.append(acc)
            
        final_mIoU = np.mean(ious)
        final_mPA = np.mean(accuracies)
        pixel_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
        
        # FWAVACC metric
        total_pixels = np.sum(cm)
        if total_pixels > 0:
            class_frequencies = cm.sum(axis=1) / total_pixels
            fwavacc = np.sum(class_frequencies * np.array(ious))
        else:
            fwavacc = 0.0

        print("\n>>>> RESULTS <<<<")
        print(f"--> Accuracy% (PA):        {pixel_accuracy * 100:.2f}%")
        print(f"--> Class Accuracy% (mPA): {final_mPA * 100:.2f}%")
        print(f"--> Mean IoU%:             {final_mIoU * 100:.2f}%")
        print(f"--> FWAVACC%:              {fwavacc * 100:.2f}%\n")

        print("--- CLASSIFICATION REPORT ---")
        print(classification_report(all_targets, all_preds, labels=unique_classes, target_names=class_names, digits=4, zero_division=0))

if __name__ == '__main__':
    evaluate()