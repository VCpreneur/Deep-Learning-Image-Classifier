# Imports
import data_helper

import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import warnings
warnings.filterwarnings("ignore")
# ----------------------------

def build_model(args, num_categories):
    if args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier.in_features
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, 512)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p = 0.5)),
                              ('fc2', nn.Linear(512, 256)),
                              ('relu2', nn.ReLU()),
                              ('drop2', nn.Dropout(p = 0.25)),
                              ('fc3', nn.Linear(256, num_categories)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        model.classifier = classifier
        model.dropout_p = [0.5, 0.25]
        model.hidden = [512, 256]

    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, 1024)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p = 0.5)),
                              ('fc2', nn.Linear(1024, 512)),
                              ('relu2', nn.ReLU()),
                              ('drop2', nn.Dropout(p = 0.25)),
                              ('fc3', nn.Linear(512, 256)),
                              ('relu3', nn.ReLU()),
                              ('drop3', nn.Dropout(p = 0.25)),
                              ('fc4', nn.Linear(256, num_categories)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        model.classifier = classifier
        model.dropout_p = [0.5, 0.25, 0.25]
        model.hidden = [1024, 512, 256]

    else:
        print("Entered architecture is not included! \nPlease insert 'densenet121' or 'vgg16'.")
        quit()

    print("Model built successfully ... \n")
    model.num_features = num_features
    return model

def evaluate_model(model, loader, criterion, device):
    loss = 0
    accuracy = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss += criterion(outputs, labels).item()

        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return loss, accuracy

def train_model(model, arch, trainloader, validloader, epochs, learning_rate, criterion, optimizer, device):
    epochs = epochs
    learning_rate = learning_rate
    model.to(device)
    steps = 0
    print_every = 20

    for e in range(epochs):

        running_loss = 0
        loss = 0

        print("Training ... ")
        model.train()

        for ii, (inputs, labels) in enumerate(trainloader):

            steps += 1
            train_accuracy = 0

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            # ------------

            # Calculate training accuracy
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            train_accuracy += equality.type(torch.FloatTensor).mean()
            # ------------

            # Backward pass
            loss.backward()
            optimizer.step()
            # ------------

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Training Accuracy: {0:.2%}".format(train_accuracy))

                running_loss = 0

        # Set network on evaluation mode for inference
        model.eval()

        print("Validating ... ")

        # Turn off gradients for validation
        with torch.no_grad():
            valid_loss, valid_accuracy = evaluate_model(model, validloader, criterion, device)

        print("--- Epoch: {} --- ".format(e+1),
              "Validation Loss: {:.3f} --- ".format(valid_loss/len(validloader)),
              "Validation Accuracy: {0:.2%} ---".format(valid_accuracy/len(validloader)))

        # Set network back on training mode
        model.train()

        model.batch_size = trainloader.batch_size
        model.epochs = epochs
        model.learning_rate = learning_rate

        return model

def print_model_details(checkpoint):
    print("---------- Model Details ----------\n")
    print("Model architecture    : {!r}".format(checkpoint['arch']))
    print("Inputs                : {!r}".format(checkpoint['input_size']))
    print("Hidden layers         : {!r}".format(checkpoint['hidden']))
    print("Dropout propabilities : {!r}".format(checkpoint['dropout_p']))
    print("Outputs               : {!r}".format(checkpoint['output_size']))
    print("Batch size            : {!r}".format(checkpoint['batch_size']))
    print("Learning rate         : {!r}".format(checkpoint['learning_rate']))
    print("Number of epochs      : {!r}".format(checkpoint['epoch']))
    print("-----------------------------------\n")


def load_saved_model(filepath, device):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121()
        model.to(device)
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden'][0])),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p = checkpoint['dropout_p'][0])),
                              ('fc2', nn.Linear(checkpoint['hidden'][0], checkpoint['hidden'][1])),
                              ('relu2', nn.ReLU()),
                              ('drop2', nn.Dropout(p = checkpoint['dropout_p'][1])),
                              ('fc3', nn.Linear(checkpoint['hidden'][1], checkpoint['output_size'])),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16()
        model.to(device)
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden'][0])),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p = checkpoint['dropout_p'][0])),
                              ('fc2', nn.Linear(checkpoint['hidden'][0], checkpoint['hidden'][1])),
                              ('relu2', nn.ReLU()),
                              ('drop2', nn.Dropout(p = checkpoint['dropout_p'][1])),
                              ('fc3', nn.Linear(checkpoint['hidden'][1], checkpoint['hidden'][2])),
                              ('relu3', nn.ReLU()),
                              ('drop3', nn.Dropout(p = checkpoint['dropout_p'][2])),
                              ('fc4', nn.Linear(checkpoint['hidden'][2], checkpoint['output_size'])),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    print_model_details(checkpoint)

    return model, checkpoint['class_to_idx'], checkpoint['cat_to_name']

def predict(image_path, model, top_k, cat_to_name, idx_to_class):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    cat_to_name = cat_to_name
    idx_to_class = idx_to_class
    model.eval()
    image = data_helper.process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor.resize_([1, 3, 224, 224])

    model.to('cpu')

    result = torch.exp(model(image_tensor))

    ps, index = result.topk(top_k)
    ps, index = ps.detach(), index.detach()

    ps.resize_([top_k])
    index.resize_([top_k])

    ps, index = ps.tolist(), index.tolist()

    top_classes = [idx_to_class[x] for x in index]
    top_categories = [cat_to_name[str(x)] for x in top_classes]

    return ps, top_categories
