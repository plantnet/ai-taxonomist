import torch.nn as nn
import torch

class TraceMobileNetV2(nn.Module):
    def __init__(self, model):
        super(TraceMobileNetV2, self).__init__()
        self.model = model
        self.feature_size = model.last_channel
        self.model.eval()

    def features(self, x):
        x = self.model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return x

    def logits(self, x):
        return self.model.classifier(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

class TraceMobileNetV3(nn.Module):
    def __init__(self, model):
        super(TraceMobileNetV3, self).__init__()
        self.model = model
        self.feature_size = model.classifier[-1].in_features
        self.model.eval()

    def features(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def logits(self, x):
        return self.model.classifier(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
