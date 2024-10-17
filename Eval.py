import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from glob import glob
import net
import time
import clip
from template import imagenet_templates
from torchvision.transforms.functional import adjust_contrast, adjust_brightness



def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default = 'content/',
                    help='File path to the content image')
parser.add_argument('--text', type=str, default="A fauvism style painting with vibrant colors",
                    help='Image resolution')
parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'experiments/decoder_iter_320000.pth')
parser.add_argument('--adaptive', type=str, default = 'experiments/adaptive_iter_320000.pth')

# Additional options
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = "output/A fauvism style painting with vibrant colors",
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args('')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
adaptive = net.adaptive
vgg = net.vgg
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

decoder.eval()
adaptive.eval()
vgg.eval()
clip_model.eval()

decoder.load_state_dict(torch.load(args.decoder))
adaptive.load_state_dict(torch.load(args.adaptive))
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
adaptive.to(device)
decoder.to(device)
clip_model.to(device)

content_tf = test_transform()
tokens = clip.tokenize(args.text).to(device)
#++++++++++++++++++++++++++++++template++++++++++++++++++++++++++++++#
#template_text = compose_text_with_templates(args.text, imagenet_templates)
#tokens = clip.tokenize(template_text).to(device)
#++++++++++++++++++++++++++++++template++++++++++++++++++++++++++++++#

#############################################################################
input_fname_pattern="*.jpg"
data = glob(os.path.join(args.content, input_fname_pattern))

if os.path.isdir(args.content):
    for i in range(len(data)):
        one_content = data[i]

        content = content_tf(Image.open(one_content))
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():

            for x in range(args.steps):

                print('iteration ' + str(x))

                Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))

                text_features = clip_model.encode_text(tokens).detach()
                text_features = text_features.mean(axis=0, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                style = adaptive(text_features.float())
                x, y = style.size()
                text_style = style.view(x, y, 1, 1)

                content = decoder(Content4_1 * text_style)
                content = adjust_contrast(content, 1.5)

                content.clamp(0, 255)

            content = content.cpu()

            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                args.output, splitext(basename(one_content))[0],
                args.text[0:20], args.save_ext
            )

            save_image(content, output_name)
