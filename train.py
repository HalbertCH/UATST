# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:01:58 2020

@author: ZJU
"""

import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image
import linecache
import random
import numpy as np
import csv

import net
from sampler import InfiniteSamplerWrapper
import clip
from template import imagenet_templates

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        #transforms.RandomCrop(256),  #256
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='../data/train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--text', type=str, default="artemis_dataset_release_v0.csv",
                    help='Image resolution')
parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=320000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--content_weight', type=float, default=0.25) #content loss
parser.add_argument('--clip_weight', type=float, default=3.0)  #p2pCLIP loss
parser.add_argument('--tv_weight', type=float, default=6e-4)  #tv loss for smoothing
parser.add_argument('--glob_weight', type=float, default=6.0)  #directional CLIP loss
parser.add_argument('--ct_weight', type=float, default=0.5)  #CLIP-based style contrastive loss
parser.add_argument('--num_crops', type=int, default=64,  #64
                    help='number of patches')
parser.add_argument('--thresh', type=float, default=0.7,
                    help='Number of domains')
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--start_iter', type=float, default=0)
args = parser.parse_args('')

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda')

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
adaptive = net.adaptive

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, clip_model, adaptive, args.start_iter)
network.train()
network.to(device)

content_tf = train_transform()
content_dataset = FlatFolderDataset(args.content_dir, content_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))

#optimizer = torch.optim.Adam([{'params': network.decoder.parameters()}], lr=args.lr)
optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.adaptive.parameters()}], lr=args.lr)

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('experiments/optimizer_iter_' + str(args.start_iter) + '.pth'))

#------------------------------source-text------------------------------#
source = "a Photo"
template_source = compose_text_with_templates(source, imagenet_templates)
tokens_source = clip.tokenize(template_source).to(device)
#------------------------------source-text------------------------------#

f = open(args.text)
csv_as_list = list(csv.reader(f, delimiter=","))
row_num = len(csv_as_list)
for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)

    # ------------------------------style-text------------------------------#
    k = random.randint(1, row_num-1)
    line = csv_as_list[k][0] + ', ' + csv_as_list[k][3]
    if len(line) > 200:
        line = line[:200]

    template_text = compose_text_with_templates(line, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)


    # negative samples
    negs = []
    for index in range(15):
        k_ = random.randint(1, row_num-1)
        while k_ == k:
            k_ = random.randint(1, row_num-1)
        neg = csv_as_list[k_][0] + ', ' + csv_as_list[k_][3]
        if len(neg) > 200:
            neg = neg[:200]
        template_text_neg = compose_text_with_templates(neg, imagenet_templates)
        tokens_neg = clip.tokenize(template_text_neg).to(device)
        negs.append(tokens_neg)
    # ------------------------------style-text------------------------------#

    img, loss_c, loss_patch, loss_glob, loss_tv, loss_ct = \
        network(content_images, tokens, negs, tokens_source, args.num_crops, args.thresh)
    loss_c = args.content_weight * loss_c
    loss_patch = args.clip_weight * loss_patch
    loss_glob = args.glob_weight * loss_glob
    loss_tv = args.tv_weight * loss_tv
    loss_ct = args.ct_weight * loss_ct
    loss = loss_c + loss_patch + loss_glob + loss_tv + loss_ct

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_patch', loss_patch.item(), i + 1)
    writer.add_scalar('loss_glob', loss_glob.item(), i + 1)
    writer.add_scalar('loss_tv', loss_tv.item(), i + 1)
    writer.add_scalar('loss_ct', loss_ct.item(), i + 1)

    ############################################################################
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if (i == 0) or ((i + 1) % 1 == 0):  # 1000
        output = torch.cat([content_images, img], 2)
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        #save_image(output, str(output_name), args.batch_size)
        save_image(output, str(output_name))

        Note = open('description.txt', mode='a')
        Note.write(str(i + 1) + ': ' + line + '\n')
        Note.close()
    ############################################################################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = adaptive.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/adaptive_iter_{:d}.pth'.format(args.save_dir,
                                                        i + 1))

        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
writer.close()
f.close()