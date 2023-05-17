#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import sys
from tqdm import tqdm
from PIL import ImageFile
import copy
import argparse
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True

import smdebug.pytorch as smd


logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion , hook):
    """
    Evaluate the performance of a given model using the provided test data.
    """
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss = 0
    running_corrects = 0
    
    total_size = len(test_loader.dataset)
    
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / total_size
    total_acc = running_corrects.double() / total_size
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, validation_loader, criterion, optimizer , eps, hook):
    """
    Train a PyTorch model using the specified data loaders, loss function, and optimizer.
    """
    epochs = eps
    best_loss = float('inf')
    image_dataset = {'train':train_loader, 'valid':validation_loader}
    loss_counter = 0
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        for phase in ['train', 'valid']:
            is_training = (phase == 'train')
            model.train(is_training)

            hook.set_mode(smd.modes.TRAIN if is_training else smd.modes.EVAL)
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                optimizer.zero_grad() if is_training else None

                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects.double() / len(image_dataset[phase])

            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                loss_counter = 0
            elif phase == 'valid':
                loss_counter += 1
            
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                                 epoch_loss,
                                                                                 epoch_acc,
                                                                                 best_loss))
            if loss_counter >= 3:
                return model

            if epoch >= 10:
                return model

    return model


def net():
    """
    Creates a ResNet-50 model pre-trained on ImageNet, with a new fully-connected layer added on top.
    """

    # load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # freeze all layers except the last fc layer
    for param in model.parameters():
        param.requires_grad = False   

    # replace the last fc layer with a new one
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 5)
    )

    # unfreeze the last fc layer
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def create_data_loaders(data, batch_size):
    """
    Create PyTorch dataloaders for training, testing, and validation datasets.
    """

    # define data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # create dataset and split into train, test, valid sets
    dataset = datasets.ImageFolder(data, transform=test_transform)
    train_set, test_set, valid_set = torch.utils.data.random_split(dataset, [5221, 2611, 2609])

    # create dataloaders with specified batch size and number of workers
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, valid_loader


def main(args):
    """Main function"""
    # Unpack input arguments for readability
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    data = args.data
    model_dir = args.model_dir
    epochs = args.epochs
    
    logger.info(f'Hyperparameters are LR: {learning_rate}, Batch Size: {batch_size}')
    logger.info(f'Data Paths: {data}')
    
    # Create dataloaders
    train_loader, test_loader, validation_loader = create_data_loaders(data, batch_size)
    
    # Initialize model and optimizer
    model = net()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    
    # Register hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    # Train model
    logger.info("Starting Model Training")
    model = train(model, train_loader, validation_loader, criterion, optimizer , epochs , hook)
    
    # Test model
    logger.info("Testing Model")
    test(model, test_loader, criterion , hook)
    
    # Save model
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, "model.pth"))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args = parser.parse_args()
    main(args)