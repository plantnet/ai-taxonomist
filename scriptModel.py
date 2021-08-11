import argparse
import json
import os
import sys
import torchvision.models as models
import torch.nn as nn
import torch
from models.tools import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Convert a trained model to TorchScript')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset', required=True)
# parser.add_argument("--trace", type=str, help="Exported TorchScript weights .pt", default=None)
parser.add_argument("--snapshot", type=str, help="Input snapshot.pth.tar", default=None)
# parser.add_argument('--crop-size', default=224, type=int,
#                     help='Network input image size')


def main():
    args = parser.parse_args()

    args.trace = os.path.join(args.data, 'network', 'network.pt')
    if not args.snapshot:
        args.snapshot = os.path.join(args.data, 'network', 'model_best.pth.tar')

    # load snapshot
    if not os.path.isfile(args.snapshot):
        print('Failed to load snapshot', args.snapshot)
        sys.exit(-1)

    print("=> loading checkpoint '{}'".format(args.snapshot))
    device = torch.device('cpu')
    checkpoint = torch.load(args.snapshot, map_location=device)
    args.arch = checkpoint['arch']
    args.num_classes = checkpoint['num_classes']
    args.size = checkpoint['crop_size']
    # get rid of ddp
    state_dict = dict()
    for k,v in checkpoint['state_dict'].items():
        if k.startswith('module.'):
            state_dict[k[len('module.'):]] = v
        else:
            state_dict[k.removeprefix('module.')] = v

    print("=> creating model '{}'".format(args.arch))
    base_model = models.__dict__[args.arch](num_classes=args.num_classes)
    base_model.load_state_dict(state_dict)
    model = traceable_module(base_model)
    if not model:
        print("=> model architecture '{}' is not supported".format(args.arch))
        sys.exit(-2)

    print("=> tracing model")
    input = torch.rand(1, 3, args.size, args.size)
    features = model.features(input)
    traced = torch.jit.trace_module(model, {'features': input, 'logits': features})
    print("=> saving torchscript model '{}'".format(args.trace))
    traced.save(args.trace)
    with open(os.path.join(args.data, 'network', 'network.json'), 'w') as f:
        json.dump({'network':'network.pt', 'features':'features', 'logits':'logits', 'feat_size': model.feature_size,
                   'num_classes': args.num_classes, 'arch':args.arch, 'img_size':args.size}, f)

if __name__ == '__main__':
    main()


