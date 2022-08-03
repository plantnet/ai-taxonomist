"""
Helpers to allow to extract features from the given models
"""
from .MobileNet import *
from .ResNet import *
from .Inception import *
from .DenseNet import *

import torch.nn as nn
import torchvision

def reset_classifier(model: nn.Module, num_classes: int) -> int:
    feat_size = -1
    if isinstance(model, torchvision.models.MobileNetV2) \
            or isinstance(model, torchvision.models.MobileNetV3):
        feat_size = model.classifier[-1].in_features
        linear = nn.Linear(feat_size, num_classes)
        nn.init.normal_(linear.weight, 0, 0.01)
        nn.init.zeros_(linear.bias)
        model.classifier[-1] = linear
    elif isinstance(model,torchvision.models.ResNet):
        feat_size = model.fc.in_features
        linear = nn.Linear(feat_size, num_classes)
        nn.init.normal_(linear.weight, 0, 0.01)
        nn.init.zeros_(linear.bias)
        model.fc = linear
    elif isinstance(model, torchvision.models.Inception3):
        feat_size = model.fc.in_features
        linear = nn.Linear(feat_size, num_classes)
        nn.init.normal_(linear.weight, 0, 0.01)
        nn.init.zeros_(linear.bias)
        model.fc = linear
        # also reset aux classifier
        if model.AuxLogits:
            aux_feat_size = model.AuxLogits.fc.in_features
            linear = nn.Linear(aux_feat_size, num_classes)
            nn.init.normal_(linear.weight, 0, 0.01)
            nn.init.zeros_(linear.bias)
            model.AuxLogits.fc = linear
    elif isinstance(model, torchvision.models.DenseNet):
        feat_size = model.classifier.in_features
        linear = nn.Linear(feat_size, num_classes)
        nn.init.normal_(linear.weight, 0, 0.01)
        nn.init.zeros_(linear.bias)
        model.classifier = linear
    return feat_size

def traceable_module(model: nn.Module) -> nn.Module:
    if isinstance(model, torchvision.models.MobileNetV2):
        return TraceMobileNetV2(model)
    elif isinstance(model, torchvision.models.MobileNetV3):
        return TraceMobileNetV3(model)
    elif isinstance(model,torchvision.models.ResNet):
        return TraceResNet(model)
    elif isinstance(model, torchvision.models.Inception3):
        return TraceInceptionV3(model)
    elif isinstance(model, torchvision.models.DenseNet):
        return TraceDenseNet(model)

    return None



