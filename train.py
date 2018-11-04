# Imports
import argparse
import data_helper
import model_helper

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import warnings
warnings.filterwarnings("ignore")
# ----------------------------

parser = argparse.ArgumentParser(description='Build, train and save image classifier model')

parser.add_argument('data_dir',
                    #'--data_dir',
                    #action = 'store',
                    #dest = 'data_dir',
                    nargs='?',
                    type = str,
                    default = 'flowers',
                    help='directory of the dataset')

parser.add_argument('--gpu',
                    action='store_true',
                    dest = 'gpu',
                    default = False,
                    help = 'enable GPU mode')

parser.add_argument('--epochs',
                    action='store',
                    dest = 'epochs',
                    type = int,
                    default = 1,
                    help = 'number of epochs')

parser.add_argument('--arch',
                    action = 'store',
                    dest = 'arch',
                    type = str,
                    choices=["densenet121", "vgg16"],
                    default = 'densenet121',
                    help = 'pre-trained model architecture: densenet121 or vgg16')

parser.add_argument('--learning_rate',
                    action = 'store',
                    dest = 'learning_rate',
                    type = float,
                    default = 0.001,
                    help = 'learning rate')

args = parser.parse_args()
print("\n")
print("--------- Entered Arguments ----------")
print("Data directory     : {!r}".format(args.data_dir))
print("Enable gpu         : {!r}".format(args.gpu))
print("Epoch(s)           : {!r}".format(args.epochs))
print("Model architecture : {!r}".format(args.arch))
print("Learning rate      : {!r}".format(args.learning_rate))
print("--------------------------------------\n")

if __name__ == "__main__":
    device = torch.device("cuda:0" if args.gpu else "cpu")

    trainloader, validloader, testloader, num_categories, cat_to_name, class_to_idx = data_helper.load_imgs(args.data_dir)

    model = model_helper.build_model(args, num_categories)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model = model_helper.train_model(model,
                                     args.arch,
                                     trainloader, validloader,
                                     args.epochs,
                                     args.learning_rate,
                                     criterion,
                                     optimizer,
                                     device)

    model.eval()
    print("\nTesting ... ")
    # Turn off gradients for validation
    with torch.no_grad():
        test_loss, test_accuracy = model_helper.evaluate_model(model, testloader, criterion, device)

    print("Testing Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Testing Accuracy: {0:.2%} \n".format(test_accuracy/len(testloader)))

    # Save model
    print("\nSaving checkpoint ... \n")
    checkpoint = {'arch': args.arch,
                  'input_size': model.num_features,
                  'hidden': model.hidden,
                  'output_size': num_categories,
                  'batch_size': model.batch_size,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': class_to_idx,
                  'epoch': model.epochs,
                  'dropout_p': model.dropout_p,
                  'learning_rate': model.learning_rate,
                  'cat_to_name': cat_to_name,
                 }

    model_helper.print_model_details(checkpoint)
    
    torch.save(checkpoint, 'my_checkpoint.pth')
    print("Model saved successfully.\n")
