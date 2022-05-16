from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from torchvision.io import read_image
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



class WikiArts(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        ''''
        img_dir: images are stored in a directory img_dir
        annotations_file: labels are stored separately in a pickle file
        '''
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # print(self.img_labels.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class Model(nn.Module):
    def __init__(self, mod, path_train, path_valid, path_test, num_classes, val =True):
        '''
        Input:
            val: bool; True means that pre-trained model will be used
        '''
        super(Model, self).__init__()
        self.train_set = torch.load(path_train)
        self.valid_set = torch.load(path_valid)
        self. test_set = torch.load(path_test)
        self.val = val
        self.num_classes = num_classes
        self.mod = mod
        # self.mod = torch.hub.load(
        #     'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=self.val)
        # self.mod.fc = nn.Linear(self.mod.fc.in_features, num_classes)
    
   
    def train(self, method = 'SGD', num_epochs = 10, learning_rate = 0.001):
        if method == 'SGD':
            BEST_MODEL_PATH = './Preparing_devOps/best_model.pth'
            BEST_OPTIMIZER_PATH = './Preparing_devOps/best_optim.pth'
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.mod.parameters()), lr=learning_rate)

            best_accuracy = 0.0
            train_losses = []
            valid_losses = []
            valid_accuracies = []

            for epoch in range(num_epochs):
                train_loss = 0.0
                valid_loss = 0.0

                # Train set
                self.mod.train()
                # i = 0
                for images, labels in iter(self.train_set):
                    optimizer.zero_grad()
                    outputs = self.mod(images)
                    labels=torch.from_numpy(
                        np.array([labels[i] for i in range (len(labels))])).long()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * images.size(0)
                
                # Valid set
                self.mod.eval()
                valid_error_count = 0.0
                for data, target in self.valid_set:     
                    outputs = self.mod(images)
                    loss = F.cross_entropy(outputs, labels)
                    valid_loss += loss.item() * images.size(0)
                    valid_error_count += float(len(labels[labels != outputs.argmax(1)]))

                # Comparison valid and test set for each epoch
                train_loss = train_loss/len(self.train_set.sampler)
                valid_loss = valid_loss/len(self.valid_set.sampler)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                # Calculate accuracy for validation dataset
                validation_accuracy = 1.0 - float(valid_error_count) / float(len(self.valid_set))
                valid_accuracies.append(validation_accuracy)
                print('%d: %f' % (epoch, validation_accuracy))
                if validation_accuracy >= best_accuracy:
                    torch.save(self.mod.state_dict(), BEST_MODEL_PATH)
                    torch.save(optimizer.state_dict(), BEST_OPTIMIZER_PATH)
                    best_accuracy = validation_accuracy
            return train_losses, valid_losses, valid_accuracies
                        
                    
    