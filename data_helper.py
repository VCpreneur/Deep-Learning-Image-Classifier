# Imports
import numpy as np
import json
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import warnings
warnings.filterwarnings("ignore")
# ----------------------------

def load_imgs(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           #transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = dict()
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define the dataloaders using the image datasets and the trainforms
    batch_size = 64
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    print ("Data loaded successfully ...\n")

    # Explore the data
    print('Number of data points:')
    print('\t' + 'Training dataset: {}'.format(len(train_data)))
    print('\t' + 'Validation dataset: {}'.format(len(valid_data)))
    print('\t' + 'Testing dataset: {}\n'.format(len(test_data)))

    class_to_idx = train_data.class_to_idx
    categories = train_data.classes
    num_categories = len(categories)
    print ("Number of categories: {}\n".format(num_categories))

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return trainloader, validloader, testloader, num_categories, cat_to_name, class_to_idx

# --------------------

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img_loader = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    image = Image.open(image)
    image = img_loader(image).float()

    image = np.array(image)

    return image
