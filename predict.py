# Imports
import data_helper
import model_helper

import argparse
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import warnings
warnings.filterwarnings("ignore")
# ----------------------------

parser = argparse.ArgumentParser(description='Predict image category using a saved model checkpoint')

parser.add_argument('img_path',
                    #'--data_dir',
                    #action = 'store',
                    #dest = 'data_dir',
                    nargs='?',
                    type = str,
                    default = 'flowers/test/1/image_06743.jpg',
                    help='path of image to classify')

parser.add_argument('checkpoint',
                    #'--checkpoint',
                    #action = 'store',
                    #dest = 'checkpoint',
                    nargs='?',
                    type = str,
                    default = 'my_checkpoint.pth',
                    help='name of saved model checkpoint file')

parser.add_argument('--gpu',
                    action='store_true',
                    dest = 'gpu',
                    default = False,
                    help = 'enable GPU mode')

parser.add_argument('--top_k',
                    action = 'store',
                    dest = 'top_k',
                    type = float,
                    default = 5,
                    help = 'top K most likely classes')

args = parser.parse_args()
print("\n")
print("---------Entered Arguments----------")
print("Image path         = {!r}".format(args.img_path))
print("Checkpoint         = {!r}".format(args.checkpoint))
print("Enable gpu         = {!r}".format(args.gpu))
print("Top K              = {!r}".format(args.top_k))
print("------------------------------------\n")

print("Loading saved model ...\n")

device = torch.device("cuda:0" if args.gpu else "cpu")
model, class_to_idx, cat_to_name = model_helper.load_saved_model(args.checkpoint, device)
idx_to_class = { v : k for k,v in class_to_idx.items()}

print("Predicting ...\n")

ps, top_categories = model_helper.predict(args.img_path, model, args.top_k, cat_to_name, idx_to_class)
print('Results of predicting image category for ("{}"):'.format(args.img_path.split('/')[-1]))
print('Top {} Categories: '.format(args.top_k), '\n', '\t', top_categories)
print('Top {} Probabilities: '.format(args.top_k), '\n', '\t', ps)
print("\nPrediction completed successfully.\n")
