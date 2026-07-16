import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler 
import numpy as np
import sys
import os
from dotenv import load_dotenv
import argparse
import re
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import CulturalDataset
from models.networks import ResnetGenerator, PatchSampleF, NLayerDiscriminator, GANLoss
from utility.osudl_loss import OSUDLLossManager
from models.PSPNet import PSPNet 

def calculate_iou(pred, target, num_classes, ignore_index=255):
    pred_classes = torch.argmax(pred, dim=1)
    iou_list = []
    
    for cls in range(num_classes):
        if cls == ignore_index: 
            continue
        
        pred_inds = (pred_classes == cls)
        target_inds = (target == cls)
        
        intersection = (pred_inds & target_inds).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union > 0:
            iou_list.append(float(intersection) / float(union))
            
    return sum(iou_list) / len(iou_list) if iou_list else 0.0

def calculate_inverse_frequency_weights(dataloader, num_classes, device, num_batches_to_scan=200):
    print(f"Calculating Inverse Frequency weights on {num_batches_to_scan} batches...")
    counts = torch.zeros(num_classes)
    
    for i, (_, labels) in enumerate(dataloader):
        if i >= num_batches_to_scan:
            break
        mask = labels != 255
        if mask.any():
            bincount = torch.bincount(labels[mask].flatten(), minlength=num_classes)
            counts += bincount.float()
            
    counts[counts == 0] = counts.mean() if counts.mean() > 0 else 1.0
    total_pixels = counts.sum()
    freqs = counts / total_pixels
    weights = 1.0 / torch.log(1.02 + freqs)
    weights[0] *= 0.1
    weights = (weights / weights.sum()) * num_classes
    print("Weights successfully calculated!")
    return weights.to(device)

def resume_training(psp_net, netG, real_perc, device):
    start_epoch = 0
    if os.path.exists("weights"):
        saved_epochs = []
        for f in os.listdir("weights"):
            match = re.search(rf'pspnet_epoch_(\d+)_real{int(real_perc*100)}\.pth', f)
            if match:
                saved_epochs.append(int(match.group(1)))
        if saved_epochs:
            start_epoch = max(saved_epochs)
            print(f"[AUTO-RESUME] Loading weights from epoch {start_epoch}...")
            psp_net.load_state_dict(torch.load(f"weights/pspnet_epoch_{start_epoch}_real{int(real_perc*100)}.pth", map_location=device))
            netG.load_state_dict(torch.load(f"weights/generator_epoch_{start_epoch}_real{int(real_perc*100)}.pth", map_location=device))
    return start_epoch

def evaluate_model(psp_net, loader_real_val, device, num_classes, real_perc):
    print("\n" + "="*50)
    print(" STARTING FINAL EVALUATION ON REAL DATA ")
    print("="*50)
    
    psp_net.eval() 
    all_preds, all_targets = [], []

    if len(loader_real_val) == 0:
        print("WARNING: Validation dataset is empty.")
        return

    with torch.no_grad():
        for img_real, label_real in loader_real_val:
            img_real, label_real = img_real.to(device), label_real.to(device)
            label_real[(label_real >= num_classes) & (label_real != 255)] = 255

            img_real_psp = F.interpolate(img_real, size=(473, 473), mode='bilinear', align_corners=True)
            outputs_psp = psp_net(img_real_psp)
            outputs = F.interpolate(outputs_psp, size=label_real.shape[1:], mode='bilinear', align_corners=True)
            
            pred_classes = torch.argmax(outputs, dim=1)
            
            pred_flat = pred_classes.cpu().numpy().flatten()
            label_flat = label_real.cpu().numpy().flatten()
            
            mask = (label_flat != 255)
            all_preds.extend(pred_flat[mask])
            all_targets.extend(label_flat[mask])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print(f"\n>>>> FINAL RESULTS (MIXED REAL {real_perc*100}%) <<<<")
    
    if len(all_targets) == 0:
        print("WARNING: No valid pixels found.")
        return

    unique_classes = np.arange(num_classes)
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
    
    total_pixels = np.sum(cm)
    fwavacc = np.sum((cm.sum(axis=1) / total_pixels) * np.array(ious)) if total_pixels > 0 else 0.0

    print(f"--> Accuracy% (PA):        {pixel_accuracy * 100:.2f}%")
    print(f"--> Class Accuracy% (mPA): {final_mPA * 100:.2f}%")
    print(f"--> Mean IoU%:             {final_mIoU * 100:.2f}%")
    print(f"--> FWAVACC%:              {fwavacc * 100:.2f}%\n")
    print("--- CLASSIFICATION REPORT ---")
    print(classification_report(all_targets, all_preds, labels=unique_classes, target_names=class_names, digits=4, zero_division=0))

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = args.dataset_path
    
    dataset_synth = CulturalDataset(dataset_path, domain='synthetic', split='train')
    dataset_real = CulturalDataset(dataset_path, domain='real', split='train', percentage=args.real_perc)

    loader_synth = DataLoader(dataset_synth, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    real_sampler = RandomSampler(dataset_real, replacement=True, num_samples=len(dataset_synth))
    loader_real = DataLoader(dataset_real, batch_size=args.batch_size, sampler=real_sampler, drop_last=True, num_workers=args.num_workers, pin_memory=True)

    dataset_real_val = CulturalDataset(dataset_path, domain='real', split='val')
    if len(dataset_real_val) == 0:
        dataset_real_val = CulturalDataset(dataset_path, domain='real', split='test')
    loader_real_val = DataLoader(dataset_real_val, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    netG = ResnetGenerator(input_nc=3, output_nc=3).to(device)
    netD = NLayerDiscriminator(input_nc=3).to(device)
    netF_q = PatchSampleF(nc=256).to(device)
    netF_k = PatchSampleF(nc=256).to(device)
    psp_net = PSPNet(layers=50, classes=args.num_classes).to(device)

    optimizer_G = torch.optim.Adam(list(netG.parameters()) + list(netF_q.parameters()) + list(netF_k.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_seg = torch.optim.Adam(psp_net.parameters(), lr=0.0001)

    start_epoch = 0
    if args.resume:
        start_epoch = resume_training(psp_net, netG, args.real_perc, device)
            
    if start_epoch > 0:
        for param_group in optimizer_G.param_groups: param_group['initial_lr'] = 0.0002
        for param_group in optimizer_D.param_groups: param_group['initial_lr'] = 0.0002
        for param_group in optimizer_seg.param_groups: param_group['initial_lr'] = 0.0001
        
    last_ep = start_epoch - 1 if start_epoch > 0 else -1

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs, last_epoch=last_ep)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs, last_epoch=last_ep)
    scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=args.epochs, last_epoch=last_ep)

    loss_manager = OSUDLLossManager(device)
    criterionGAN = GANLoss('lsgan').to(device)
    class_weights = calculate_inverse_frequency_weights(loader_synth, args.num_classes, device)
    criterionSeg = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    print(f"\nJoint Training (OS-DCLGAN + PSPNet) - MIXED REAL = {args.real_perc*100}%")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        if epoch < args.warmup_epochs:
            print("-> [Warm-up Phase] Segmentation network learns ONLY from original synthetic data.")
        else:
            print("-> [Domain Adaptation] Segmentation network learns from synthetic, translated (and real if supervised).")
        
        epoch_iou = 0.0
        total_batches = 0
        
        netG.train()
        netD.train()
        psp_net.train()
        
        for i, ((img_synth, label_synth), (img_real, label_real)) in enumerate(zip(loader_synth, loader_real)):

            img_synth, label_synth = img_synth.to(device), label_synth.to(device)
            img_real, label_real = img_real.to(device), label_real.to(device)

            label_synth[(label_synth >= args.num_classes) & (label_synth != 255)] = 255
            label_real[(label_real >= args.num_classes) & (label_real != 255)] = 255

            optimizer_G.zero_grad()
            optimizer_seg.zero_grad()

            # --- GENERATOR GAN ---
            fake_real = netG(img_synth)
            pred_fake = netD(fake_real)
            loss_G_GAN = criterionGAN(pred_fake, target_is_real=True)
            loss_NCE = loss_manager.calculate_NCE_loss(img_synth, fake_real, netG, netG, netF_q, netF_k)
            loss_G_total = loss_G_GAN + (loss_NCE * 2.0)

            # --- SEGMENTATION FAKE ---
            fake_real_psp = F.interpolate(fake_real.detach(), size=(473, 473), mode='bilinear', align_corners=True)
            pred_seg_fake_main, pred_seg_fake_aux = psp_net(fake_real_psp)
            pred_seg_fake_main = F.interpolate(pred_seg_fake_main, size=label_synth.shape[1:], mode='bilinear', align_corners=True)
            pred_seg_fake_aux = F.interpolate(pred_seg_fake_aux, size=label_synth.shape[1:], mode='bilinear', align_corners=True)
            
            # --- SEGMENTATION REAL ---
            img_synth_psp = F.interpolate(img_synth, size=(473, 473), mode='bilinear', align_corners=True)
            pred_seg_real_main, pred_seg_real_aux = psp_net(img_synth_psp)
            pred_seg_real_main = F.interpolate(pred_seg_real_main, size=label_synth.shape[1:], mode='bilinear', align_corners=True)
            pred_seg_real_aux = F.interpolate(pred_seg_real_aux, size=label_synth.shape[1:], mode='bilinear', align_corners=True)
            
            # --- CALC SEGMENTATION LOSS ---
            valid_pixels = (label_synth != 255).sum().item()
            loss_seg = torch.tensor(0.0, device=device)
            
            if valid_pixels > 0:
                loss_seg_fake_main_val = criterionSeg(pred_seg_fake_main, label_synth)
                loss_seg_fake_aux_val = criterionSeg(pred_seg_fake_aux, label_synth)
                loss_seg_fake = loss_seg_fake_main_val + (0.4 * loss_seg_fake_aux_val)
                
                loss_seg_real_main_val = criterionSeg(pred_seg_real_main, label_synth)
                loss_seg_real_aux_val = criterionSeg(pred_seg_real_aux, label_synth)
                loss_seg_real = loss_seg_real_main_val + (0.4 * loss_seg_real_aux_val)

                loss_seg_real_domain = 0.0
                if args.real_perc > 0.0 and args.supervised and (label_real != 255).any():
                    img_real_psp = F.interpolate(img_real, size=(473, 473), mode='bilinear', align_corners=True)
                    pred_seg_real_domain_main, pred_seg_real_domain_aux = psp_net(img_real_psp)
                    pred_seg_real_domain_main = F.interpolate(pred_seg_real_domain_main, size=label_real.shape[1:], mode='bilinear', align_corners=True)
                    pred_seg_real_domain_aux = F.interpolate(pred_seg_real_domain_aux, size=label_real.shape[1:], mode='bilinear', align_corners=True)
                    
                    loss_domain_main = criterionSeg(pred_seg_real_domain_main, label_real)
                    loss_domain_aux = criterionSeg(pred_seg_real_domain_aux, label_real)
                    loss_seg_real_domain = loss_domain_main + (0.4 * loss_domain_aux)
                    
                if epoch < args.warmup_epochs:
                    loss_seg = loss_seg_real
                else:
                    if args.real_perc > 0.0 and args.supervised and (label_real != 255).any():
                        loss_seg = (loss_seg_fake + loss_seg_real + loss_seg_real_domain) / 3.0
                    else:
                        loss_seg = (loss_seg_fake + loss_seg_real) * 0.5


            # --- WEIGHT UPDATES ---
            if not torch.isnan(loss_G_total) and not torch.isinf(loss_G_total):
                loss_G_total.backward()
                torch.nn.utils.clip_grad_norm_(list(netG.parameters()) + list(netF_q.parameters()) + list(netF_k.parameters()), max_norm=10.0)
                optimizer_G.step()
            
            optimizer_seg.zero_grad() 
            
            if loss_seg.item() > 0 and not torch.isnan(loss_seg) and not torch.isinf(loss_seg):
                loss_seg.backward()
                torch.nn.utils.clip_grad_norm_(psp_net.parameters(), max_norm=10.0)
                optimizer_seg.step()
            
            batch_iou = calculate_iou(pred_seg_real_main, label_synth, num_classes=args.num_classes)
            epoch_iou += batch_iou
            total_batches += 1

            # --- DISCRIMINATOR ---
            optimizer_D.zero_grad()
            pred_real_D = netD(img_real)
            loss_D_real = criterionGAN(pred_real_D, target_is_real=True)
            pred_fake_D = netD(fake_real.detach())
            loss_D_fake = criterionGAN(pred_fake_D, target_is_real=False)
            loss_D_total = (loss_D_real + loss_D_fake) * 0.5
            
            if not torch.isnan(loss_D_total) and not torch.isinf(loss_D_total):
                loss_D_total.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=10.0)
                optimizer_D.step()

            if i % 50 == 0:
                print(f"Batch {i} -> Loss D: {loss_D_total.item():.4f} | Loss G: {loss_G_total.item():.4f} | Loss Seg: {loss_seg.item():.4f} | Batch IoU: {batch_iou:.4f}")

        scheduler_G.step()
        scheduler_D.step()
        scheduler_seg.step()

        os.makedirs("weights", exist_ok=True)
        torch.save(psp_net.state_dict(), f"weights/pspnet_epoch_{epoch+1}_real{int(args.real_perc*100)}.pth")
        torch.save(netG.state_dict(), f"weights/generator_epoch_{epoch+1}_real{int(args.real_perc*100)}.pth")

        print(f">> END OF EPOCH {epoch+1} <<")
        if total_batches > 0:
            print(f"Mean Training IoU: {epoch_iou / total_batches:.4f}")

    print("\nSaving final weights...")
    torch.save(psp_net.state_dict(), f"weights/pspnet_final_real{int(args.real_perc*100)}.pth")
    torch.save(netG.state_dict(), f"weights/generator_final_real{int(args.real_perc*100)}.pth")

    evaluate_model(psp_net, loader_real_val, device, args.num_classes, args.real_perc)

if __name__ == '__main__':
    load_dotenv()

    dataset_path = os.getenv('DATA_PATH', './EGO-CH-OBJ-SEG/EGO-CH-OBJ-SEG')

    parser = argparse.ArgumentParser(description="Train OS-DCLGAN + PSPNet")
    parser.add_argument("--real_perc", type=float, default=1.0, help="Percentage of real data to use (0.0 to 1.0)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest saved epoch")
    parser.add_argument("--supervised", action="store_true", help="Use real labels for domain adaptation")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of warmup epochs for PSPNet")
    parser.add_argument("--num_classes", type=int, default=25, help="Number of segmentation classes")
    parser.add_argument("--dataset_path", type=str, default=dataset_path, help="Path to the dataset")
    
    args = parser.parse_args()
    train(args)