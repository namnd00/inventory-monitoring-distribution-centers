
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import time

import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion):
    """Test model function."""
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects.double() // len(test_loader)
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, validation_loader, criterion, optimizer, max_epochs):
    """Train model function."""
    best_loss = float('inf')
    loss_counter = 0
    
    for epoch in range(max_epochs):
        logger.info(f"Epoch: {epoch}")
        
        # Train and validation phases
        for phase in ['train', 'valid']:
            is_training = phase == 'train'
            model.train(is_training)
            data_loader = train_loader if is_training else validation_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if is_training:
                        loss.backward()
                        optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects / len(data_loader.dataset)

            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                                 epoch_loss,
                                                                                 epoch_acc,
                                                                                 best_loss))

            # Early stopping
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                loss_counter = 0
            elif phase == 'valid':
                loss_counter += 1
                
            if loss_counter >= 3:
                logger.info('Stopping early at epoch {}'.format(epoch))
                return model

    return model

    
def net():
    """Network initialization."""
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad_(False)

    # Modify last fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_ftrs, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 5))

    # Unfreeze last layer parameters
    for param in model.fc.parameters():
      param.requires_grad_(True)

    return model


def create_data_loaders(data, batch_size):
    """Create dataloaders"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data, transform=train_transform)

    # Use random_split to get train, test and validation sets
    train_set, test_set, valid_set = torch.utils.data.random_split(dataset, [5221, 2611, 2609])

    # Use DataLoader with pin_memory=True for faster data transfer to GPU
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, valid_loader


def main(args):
    """Main function."""
    # Log hyperparameters and data paths to console
    logger.info('Hyperparameters are LR: {}, Batch Size: {}'.format(args.learning_rate, args.batch_size))
    logger.info('Data Paths: {}'.format(args.data))

    # Get dataloaders with specified batch size
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)

    # Create model object and move it to device if GPU available
    model = net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train and validate model
    start_time = time.time()
    model = train(model, train_loader, validation_loader, criterion, optimizer, args.epochs)
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info('Model training completed in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    # Test model on unseen data
    test_loss, test_accuracy = test(model, test_loader, criterion)

    logger.info('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(test_loss, test_accuracy * 100))

    # Save trained model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), model_path)
    logger.info('Trained model saved at: {}'.format(model_path))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()
    main(args)