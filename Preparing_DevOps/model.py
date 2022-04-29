'''
This code is made in order to put the model into prod
'''

import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from classes import WikiArts
from torch.utils.data import DataLoader

path_labels = r'./output/path_label.csv'
path_images = r'./WikiArt_sample'


def pre_processing():
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
    data_preprocessed = WikiArts(path_labels, path_images, transform)
    return data_preprocessed

data_preprocessed = pre_processing()

def split_load_data(length_test, length_valid , num_workers, batch_size):
    train_dataset, test_dataset = torch.utils.data.random_split(
        data_preprocessed, [len(data_preprocessed) - length_test, length_test]
    )
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) - length_valid, length_valid]
    )
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def train_model(length_test = 2, length_valid =2, num_workers =4 , batch_size = 15, 
    num_classes = 4, num_epochs = 15, learning_rate = 0.001, mom = 0.9, 
):
    # Load data
    train_loader, valid_loader, test_loader = split_load_data(length_test, length_valid , num_workers, batch_size)

    # Load pre-trained model
    resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)

    # Choose optimizer
    optimizer = optim.SGD(resnet50.parameters(), lr=learning_rate, momentum=mom)

    # Train model
    BEST_MODEL_PATH = 'best_model_optmiSGD_lossCrossEntrop.pth'
    best_accuracy = 0.0
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # Train set
        resnet50.eval()
        for images, labels in iter(train_loader):
            optimizer.zero_grad()
            outputs = resnet50(images)
            labels=torch.from_numpy(
                np.array([labels[i] for i in range (len(labels))])).long()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        # Valid set
        resnet50.eval()
        valid_error_count = 0.0
        for data, target in valid_loader:      
            outputs = resnet50(images)
            loss = F.cross_entropy(outputs, labels)
            valid_loss += loss.item() * images.size(0)
            valid_error_count += float(len(labels[labels != outputs.argmax(1)]))

        # Comparison valid and test set for each epoch
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Calculate accuracy for validation dataset
        validation_accuracy = 1.0 - float(valid_error_count) / float(length_valid)
        valid_accuracies.append(validation_accuracy)
        print('%d: %f' % (epoch, validation_accuracy))
        if validation_accuracy > best_accuracy:
            torch.save(resnet50.state_dict(), BEST_MODEL_PATH)
            best_accuracy = validation_accuracy
        
        

