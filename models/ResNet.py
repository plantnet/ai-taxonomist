import torch.nn as nn
import torch


class TraceResNet(nn.Module):
    def __init__(self, model):
        super(TraceResNet, self).__init__()
        self.model = model
        self.feature_size = model.fc.in_features
        self.model.eval()

    def features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def logits(self, x):
        return self.model.fc(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
