import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from utility.cbam import CBAM 

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

# --- GENERATORE (ResNet) ---
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9):
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'reflect', norm_layer, use_dropout, use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
                      
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats # Restituisce solo le feature intermedie
            return feat, feats
        else:
            return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        p = 1 if padding_type == 'reflect' else 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0 if padding_type == 'reflect' else p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0 if padding_type == 'reflect' else p, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# --- ESTRATTORE DI FEATURE CON ATTENZIONE (Il cuore della loro innovazione) ---
class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, nc=256):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.cbam = CBAM(256, 16)
            
    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.to(feat.device)
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        self.mlp_init = True

    def qs_attn(self, feat_reshape, num_patches, B):
        feat_q = feat_reshape
        feat_k = feat_reshape.permute(0, 2, 1)
        dots = torch.bmm(feat_q, feat_k)                
        attn = dots.softmax(dim=2) 
        prob = -torch.log(attn + 1e-8)                    
        prob = torch.where(torch.isinf(prob), torch.full_like(prob, 0), prob)                      
        entropy = torch.sum(torch.mul(attn, prob), dim=2)                                        
        _, index = torch.sort(entropy)                        
        patch_id = index[:, :num_patches]    
        return attn[torch.arange(B)[:, None], patch_id, :]

    def forward(self, feats, num_patches=256, patch_ids=None, attn_mats=None):
        return_ids, return_feats, return_mats = [], [], []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
            
        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape
            if feat_id >= 2 and patch_ids is not None: 
                feat = self.cbam(feat)   
            feat_reshape = feat.permute(0,2,3,1).flatten(1,2)          
            
            if num_patches > 0:
                if feat_id < 2:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id]
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=feat.device) 
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1) 
                    attn_qs = torch.zeros(1).to(feat.device)
                else:
                    if attn_mats is not None:
                        attn_qs = attn_mats[feat_id]            
                    else:
                        attn_qs = self.qs_attn(feat_reshape, num_patches, B)                                
                    feat_reshape = torch.bmm(attn_qs, feat_reshape) 
                    x_sample = feat_reshape.flatten(0, 1) 
                    patch_id = []
            else:
                x_sample = feat_reshape
                patch_id = []
                
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
         
            return_ids.append(patch_id)
            return_mats.append(attn_qs)
            x_sample = self.l2norm(x_sample)
            return_feats.append(x_sample)

        return return_feats, return_ids, return_mats

# --- DISCRIMINATORE E LOSS GAN ---
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan'):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.gan_mode = gan_mode
        self.loss = nn.MSELoss() if gan_mode == 'lsgan' else nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)