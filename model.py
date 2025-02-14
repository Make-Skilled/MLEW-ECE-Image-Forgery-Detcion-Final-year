import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Binary classification

    def forward(self, x):
        return self.model(x)
