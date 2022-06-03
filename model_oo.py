from classes import Model
from functions import freeze_up_to_layer
import torch


# Load ResNet50 
resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)

# Freeze certain layers if necessary
resnet50 = freeze_up_to_layer(resnet50, block_number = 7)

# Define my model
mymodel = Model(resnet50, 
    'train_loader.pth', 
    'valid_loader.pth',
    'test_loader.pth',
    22
)

# Train model
train_losses, valid_losses, valid_accuracies = mymodel.train()

# Test model
mymodel.test('./model_trained')