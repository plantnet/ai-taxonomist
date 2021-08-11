import torch.nn as nn
import torch.nn.functional as F
import torch

class TraceDenseNet(nn.Module):
    def __init__(self, model):
        super(TraceDenseNet, self).__init__()
        self.model = model
        self.feature_size = model.classifier.in_features
        self.model.eval()

    def features(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def logits(self, x):
        return self.classifier(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
