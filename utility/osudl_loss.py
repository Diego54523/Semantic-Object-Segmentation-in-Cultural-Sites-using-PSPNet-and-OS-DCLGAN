import torch
from utility.patchnce import PatchNCELoss 

class OSUDLLossManager:
    def __init__(self, device, nce_layers=[4, 8, 12, 16], num_patches=256):
        self.device = device
        self.nce_layers = nce_layers
        self.num_patches = num_patches
        
        class DummyOpt:
            nce_T = 0.07
            nce_includes_all_negatives_from_minibatch = False
            batch_size = 1
        
        self.criterionNCE = [PatchNCELoss(DummyOpt()).to(self.device) for _ in nce_layers]
        self.criterionIdt = torch.nn.L1Loss().to(self.device)

    def calculate_NCE_loss(self, src, tgt, netG_encode, netG_decode, netF_q, netF_k):
        """Calcola la loss contrastiva tra l'immagine sorgente e quella tradotta"""
        n_layers = len(self.nce_layers)
        
        feat_q = netG_encode(tgt, self.nce_layers, encode_only=True)
        feat_k = netG_decode(src, self.nce_layers, encode_only=True)    
        
        feat_k_pool, sample_ids, attn_mats = netF_k(feat_k, self.num_patches, None, None)
        feat_q_pool, _, _ = netF_q(feat_q, self.num_patches, sample_ids, attn_mats)
 
        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
            
        return total_nce_loss / n_layers