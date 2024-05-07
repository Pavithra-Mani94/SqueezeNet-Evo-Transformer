# -*- coding: utf-8 -*-
"""
Created on Tue May  7 22:22:24 2024

@author: KAVITHA
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import gdown
import shutil
import numpy as np  # Import NumPy library

class ModifiedSqueezenet(nn.Module):
    def __init__(self, num_classes, dataset_url):
        super(ModifiedSqueezenet, self).__init__()
        self.dataset_url = dataset_url
        self.download_dataset()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Load the dataset
        self.dataset = ImageFolder(root='/content/drive/MyDrive/Dataset/realtime', transform=self.transform)

        # Define batch size and create data loaders
        self.batch_size = 32
        self.num_classes = num_classes

        # Define your modified Squeezenet architecture here
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # Add more layers as needed
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(96, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def download_dataset(self):
        if not os.path.exists('data'):
            os.makedirs('data')

        if not os.path.exists('data/OCT2017'):
            print("Downloading dataset from Google Drive...")
            gdown.download(self.dataset_url, 'data/OCT2017', quiet=False)
            print("Dataset downloaded successfully!")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x

    def get_dataloaders(self, train_idx, val_idx):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=val_sampler)

        return train_loader, val_loader

# Define the Google Drive URL for the dataset
google_drive_url = "https://drive.google.com/drive/folders/14VPp8FjV1NezLBMQuzZm3cGRPhAWQBCF?usp=drive_link"

# Create an instance of the ModifiedSqueezenet class
model = ModifiedSqueezenet(num_classes=4, dataset_url=google_drive_url)

# Perform 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X=np.zeros(len(model.dataset)), y=model.dataset.targets)):
    print(f'Fold {fold+1}')
    train_dataset = torch.utils.data.Subset(model.dataset, train_index)
    test_dataset = torch.utils.data.Subset(model.dataset, test_index)

    # Split train dataset into train and validation
    num_train = int(0.7 * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])

    train_loader, val_loader = model.get_dataloaders(train_dataset.indices, val_dataset.indices)
    test_loader = DataLoader(test_dataset, batch_size=model.batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training function
    def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=100):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_train_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train

            # Validation
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = running_val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

        return train_losses, val_losses, train_accuracies, val_accuracies

    # Training and Validation
    train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_loader, val_loader, optimizer, criterion)

    # Testing
    def test(model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        confusion = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)

        return accuracy, confusion, report

    accuracy, confusion, report = test(model, test_loader)
    print("Test Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

    # Plot training and validation accuracy/loss curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1))


