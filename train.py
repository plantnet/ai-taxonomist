# Inspired from https://github.com/pytorch/examples/tree/master/imagenet

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from models.tools import reset_classifier

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--data", metavar="DIR", help="path to dataset", required=True)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 2)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=100,
    type=int,
    metavar="N",
    help="print frequency (default: 100)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument(
    "--crop-size", default=224, type=int, help="Network input image size"
)
parser.add_argument(
    "--imbalanced",
    action="store_true",
    default=False,
    required=False,
    help="use weighted criterion to handle class imbalance",
)
sched_params = parser.add_argument_group(
    "reduce on plateau learning rate scheduler parameters"
)
sched_params.add_argument(
    "--factor",
    type=float,
    default=0.1,
    required=False,
    help="Factor by which the learning rate will be reduced",
)
sched_params.add_argument(
    "--patience",
    type=int,
    default=10,
    required=False,
    help="Number of epochs with no improvement after which learning rate will be reduced",
)
sched_params.add_argument(
    "--threshold",
    type=float,
    default=1e-4,
    required=False,
    help="Threshold for measuring the new optimum, to only focus on significant changes",
)
sched_params.add_argument(
    "--threshold_mode",
    type=str,
    default="rel",
    choices=["rel", "abs"],
    required=False,
    help="is the threshold relative or absolute",
)
sched_params.add_argument(
    "--cooldown",
    type=int,
    default=0,
    required=False,
    help="Number of epochs to wait before resuming normal operation after lr has been reduced",
)
sched_params.add_argument(
    "--min_lr",
    type=float,
    default=0,
    required=False,
    help="A lower bound on the learning rate of all param groups",
)
sched_params.add_argument(
    "--eps",
    type=float,
    default=1e-8,
    required=False,
    help="Minimal decay applied to lr. "
    "If the difference between new and old lr is smaller than eps, the update is ignored",
)

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # initialize pathes
    data_dir = args.data
    img_dir = os.path.join(data_dir, "img")
    train_dir = os.path.join(img_dir, "train")
    val_dir = os.path.join(img_dir, "val")

    online_dir = os.path.join(data_dir, "ai-taxonomist")
    network_dir = os.path.join(online_dir, "network")
    os.makedirs(network_dir, exist_ok=True)

    # compute number of classes
    dirs = set()
    for d in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, d)):
            dirs.add(d)
    for d in os.listdir(val_dir):
        if os.path.isdir(os.path.join(val_dir, d)):
            dirs.add(d)
    num_classes = len(dirs)

    weights = None
    if args.imbalanced:
        nb_img_per_class = []
        for c in os.listdir(train_dir):
            path = os.path.join(train_dir, c)
            if os.path.isdir(path):
                n = len(os.listdir(path))
                nb_img_per_class.append(n)
        w = [1 / math.log2(1 + n) for n in nb_img_per_class]
        weights = torch.Tensor(w)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == "inception_v3":
            model = models.__dict__[args.arch](
                pretrained=True,
                transform_input=False,
                aux_logits=True,
                init_weights=False,
            )
        else:
            model = models.__dict__[args.arch](pretrained=True)
        if reset_classifier(model, num_classes) < 0:
            print("Unsupported model", args.arch)
            sys.exit(-2)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == "inception_v3":
            model = models.__dict__[args.arch](
                num_classes=num_classes, transform_input=False
            )
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(weight=weights).cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=args.factor,
        patience=args.patience,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        cooldown=args.cooldown,
        min_lr=args.min_lr,
        eps=args.eps,
        verbose=True,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(args.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    # save class mapping
    with open(os.path.join(network_dir, "mapping.json"), "w") as f:
        json.dump(train_dataset.class_to_idx, f)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    label_xform = LabelXForm()
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose(
            [
                transforms.Resize(round(args.crop_size * 1.1)),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        target_transform=label_xform,
    )
    label_xform.configure(train_dataset.class_to_idx, val_dataset.class_to_idx)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        print("=> evaluating the model")
        validate(val_loader, model, criterion, args)
        return

    print(
        "=> training a",
        args.arch,
        "on",
        num_classes,
        "classes for",
        args.epochs,
        "epochs",
    )
    args.num_classes = num_classes
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, loss = validate(val_loader, model, criterion, args)
        scheduler.step(loss)
        print(
            "LearningRate: {epoch} {lr:.4e}".format(
                epoch=epoch, lr=scheduler._last_lr[0]
            )
        )

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "num_classes": num_classes,
                    "crop_size": args.crop_size,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                arch=args.arch,
                data_dir=network_dir,
            )


class LabelXForm:
    def __init__(self):
        self.idx = list()

    def configure(self, train_idx: dict, val_idx: dict):
        self.idx = [None] * len(val_idx)
        for clas, idx in val_idx.items():
            self.idx[idx] = train_idx[clas]

    def __call__(self, *args, **kwargs):
        if args[0] == len(self.idx):
            print("Ça chie !", args[0], ">=", len(self.idx))
        return self.idx[args[0]]


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print('rank',args.rank, 'labels', target)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output and measure accuracy
        output = model(images)
        if isinstance(output, models.InceptionOutputs):
            loss = criterion(output.logits, target) + 0.4 * criterion(
                output.aux_logits, target
            )
            acc1, acc5 = accuracy(
                output.logits, target, topk=(1, min(5, args.num_classes))
            )
        else:
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.num_classes)))

        # record loss
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.num_classes)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} ValLoss {loss.avg:.3f}".format(
                top1=top1, top5=top5, loss=losses
            )
        )

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, arch, data_dir="."):
    filename = arch + "_checkpoint.pth.tar"
    torch.save(state, data_dir + "/" + filename)
    if is_best:
        shutil.copyfile(
            data_dir + "/" + filename, data_dir + "/" + arch + "_best.pth.tar"
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
