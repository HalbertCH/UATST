# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:35:21 2020

@author: ZJU
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


cropper = transforms.Compose([
    transforms.RandomCrop(128)    # 128
])
augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

#This is our CSM module
adaptive = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
)



class Net(nn.Module):
    def __init__(self, encoder, decoder, clip_model, adaptive, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        self.decoder = decoder
        self.sm = nn.Softmax(dim=-1)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        #CLIP
        self.clip_model = clip_model
        self.adaptive = adaptive

        if(start_iter > 0):
            self.decoder.load_state_dict(torch.load('experiments/decoder_iter_' + str(start_iter) + '.pth'))
            self.adaptive.load_state_dict(torch.load('experiments/adaptive_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def clip_normalize(self, image, device):
        image = F.interpolate(image, size=224, mode='bicubic')
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

        image = (image - mean) / std
        return image

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def get_image_prior_losses(self, inputs_jit):
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

        return loss_var_l2

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        #loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
        return loss


    
    def forward(self, content, tokens, negs, tokens_source, num_crops, thresh, device='cuda'):
        content_feats = self.encode_with_intermediate(content)

        text_features = self.clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        #'adaptive' here is our CSM module
        style = self.adaptive(text_features.float())
        x, y = style.size()
        text_style = style.view(x, y, 1, 1)

        stylized = content_feats[3] * text_style
        #stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3]) + self.calc_content_loss(g_t_feats[4], content_feats[4])   # True; True

        # style_patch loss and direction loss
        text_source = self.clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)

        text_direction = text_features - text_source
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        # global style direction loss
        source_features = self.clip_model.encode_image(self.clip_normalize(content, device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

        glob_features = self.clip_model.encode_image(self.clip_normalize(g_t, device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

        glob_direction = glob_features - source_features
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

        loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

        #######################################################################################################
        # patch style direction loss
        # loss_patch = 0
        img_proc = []
        photo_proc = []
        for n in range(num_crops):
            seed = torch.random.seed()

            torch.random.manual_seed(seed)
            target_crop = cropper(g_t)
            torch.random.manual_seed(seed)
            source_crop = cropper(content)
            photo_proc.append(source_crop)

            target_crop = augment(target_crop)
            img_proc.append(target_crop)

        img_proc = torch.cat(img_proc, dim=0)
        img_aug = img_proc
        # save_image(img_aug, 'cs.jpg')

        photo_proc = torch.cat(photo_proc, dim=0)
        # save_image(photo_proc, 'c.jpg')

        image_features = self.clip_model.encode_image(self.clip_normalize(img_aug, device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

        patch_features = self.clip_model.encode_image(self.clip_normalize(photo_proc, device))
        patch_features /= (patch_features.clone().norm(dim=-1, keepdim=True))

        img_direction = image_features - patch_features
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        #loss_patch = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1)).mean()
        loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss_temp[loss_temp < thresh] = 0
        loss_patch = loss_temp.mean()
        #########################################################################################################

        loss_tv = self.get_image_prior_losses(g_t)

        ####################################################################################################

        text_comparisons = text_features
        for i in range(len(negs)):
            negs_features = self.clip_model.encode_text(negs[i]).detach()
            negs_features = negs_features.mean(axis=0, keepdim=True)
            negs_features /= negs_features.norm(dim=-1, keepdim=True)
            text_comparisons = torch.cat([text_comparisons, negs_features], 0)

        loss_contrastive = self.compute_contrastive_loss(glob_features, text_comparisons, 0.2, 0)

        #loss_contrastive = loss_tv
        ####################################################################################################

        return g_t, loss_c, loss_patch, loss_glob, loss_tv, loss_contrastive