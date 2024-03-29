import csv
import datetime
import os
import sys
import pip

import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast

from models import *
from tinyimagenet_module import TinyImageNet
from learning_module import TINYIMAGENET_ROOT, PoisonedDataset, get_transform, now

from typing import List
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, NormalizeImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

WRITE_PATH = './ffcv_files'

data_mean_std_dict = {
    "cifar10": (np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))),
    "cifar100": (np.array((0.5071, 0.4867, 0.4408)), np.array((0.2675, 0.2565, 0.2761))),
    "tinyimagenet_all": (np.array((0.4802, 0.4481, 0.3975)), np.array((0.2302, 0.2265, 0.2262))),
    "tinyimagenet_first": (np.array((0.4802, 0.4481, 0.3975)), np.array((0.2302, 0.2265, 0.2262))),
    "tinyimagenet_last": (np.array((0.4802, 0.4481, 0.3975)), np.array((0.2302, 0.2265, 0.2262))),
}

data_mean_std_dict_255 = {
    "cifar10": (255*data_mean_std_dict['cifar10'][0], 255*data_mean_std_dict['cifar10'][1]),
    "cifar100": (255*data_mean_std_dict['cifar100'][0], 255*data_mean_std_dict['cifar100'][1]),
    "tinyimagenet_all": (255*data_mean_std_dict['tinyimagenet_all'][0], 255*data_mean_std_dict['tinyimagenet_all'][1]),
    "tinyimagenet_first": (255*data_mean_std_dict['cifar10'][0], 255*data_mean_std_dict['cifar10'][1]),
    "tinyimagenet_last": (255*data_mean_std_dict['cifar10'][0], 255*data_mean_std_dict['cifar10'][1]),
}

model_paths = {
    "cifar10": {
        "whitebox": "pretrained_models/ResNet18_CIFAR100.pth",
        "blackbox": [
            "pretrained_models/VGG11_CIFAR100.pth",
            "pretrained_models/MobileNetV2_CIFAR100.pth"
        ],
    },
    "tinyimagenet_last": {
        "whitebox": "pretrained_models/VGG16_Tinyimagenet_first.pth",
        "blackbox": [
            "pretrained_models/ResNet34_Tinyimagenet_first.pth",
            "pretrained_models/MobileNetV2_Tinyimagenet_first.pth",
        ],
    },
}

def write_ffcv(name, set_type, dataset, args):
    """
    Write {name} (cifar10, tinyimagenet_first, etc.) dataset of type set_type (train or test) to FFCV dataset. 
    """
    if set_type == 'train':
        writer = DatasetWriter(f"{WRITE_PATH}/{name}_{set_type}_{args.poisons_path.split('/')[1]}.beton", 
                                {'image': RGBImageField(), 'label': IntField(), 'indicator': IntField()})
    else:
        writer = DatasetWriter(f"{WRITE_PATH}/{name}_{set_type}_{args.poisons_path.split('/')[1]}.beton",
                                {'image': RGBImageField(), 'label': IntField()})
    writer.from_indexed_dataset(dataset)

def get_pipeline(normalize, augment, dataset="CIFAR10", device = 'cuda'):
    """Function to perform required transformation on the tensor
    input:
        normalize:      Bool value to determine whether to normalize data
        augment:        Bool value to determine whether to augment data
        dataset:        Name of the dataset
    return
        Pytorch tranforms.Compose with list of all transformations
    """

    dataset = dataset.lower()
    mean, std = data_mean_std_dict_255[dataset]
    if "tinyimagenet" in dataset:
        dataset = "tinyimagenet"
    cropsize = {"cifar10": 32, "cifar100": 32, "tinyimagenet": 64}[dataset]
    padding = 4

    pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    if normalize and augment:
        # transform_list = [
        #     transforms.RandomCrop(cropsize, padding = padding), 
        #     RandomHorizontalFlip(), 
        #     ToTensor(),
        #     ToDevice(device, non_blocking = True),
        #     ToTorchImage(),
        #     transforms.Normalize(mean, std)
        # ]

        # Ignoring RandomCrop for now. 
        transform_list = [RandomHorizontalFlip(),
                            ToTensor(),
                            ToDevice(device, non_blocking = True),
                            ToTorchImage(),
			                Convert(torch.float16),
                            transforms.Normalize(mean, std)]
    elif augment:
        # transform_list = [
        #     transforms.RandomCrop(cropsize, padding = padding),
        #     RandomHorizontalFlip(),
        #     ToTensor(),
        #     ToDevice(device, non_blocking = True),
        #     ToTorchImage()
        # ]
        transform_list = [RandomHorizontalFlip(),
                            ToTensor(),
                            ToDevice(device, non_blocking = True),
                            ToTorchImage(),
                            Convert(torch.float16)]

    elif normalize:
        transform_list = [ToTensor(), ToDevice(device, non_blocking = True), ToTorchImage(), Convert(torch.float16), transforms.Normalize(mean, std)]
    else:
        transform_list = [ToTensor(), ToDevice(device, non_blocking = True), ToTorchImage(), Convert(torch.float16)]

    pipeline.extend(transform_list)

    return pipeline

def get_dataset(args, poison_tuples, poison_indices, device = 'cuda', simple = False):
    if args.dataset.lower() == "cifar10":
        transform_train = get_pipeline(args.normalize, args.train_augment, device = device)
        transform_test = get_pipeline(args.normalize, False, device = device)
        old_transform_test = get_transform(args.normalize, False)
        cleanset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True)
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True)
        
        dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.ToTensor())
        
        num_classes = 10

    elif args.dataset.lower() == "tinyimagenet_first":
        transform_train = get_pipeline(
            args.normalize, args.train_augment, dataset=args.dataset, device = device
        )
        transform_test = get_pipeline(args.normalize, False, dataset=args.dataset, device = device)
        old_transform_test = get_transform(args.normalize, False, dataset=args.dataset)
        cleanset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            classes="firsthalf"
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            classes="firsthalf"
        )
        dataset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transforms.ToTensor(),
            classes="firsthalf"
        )
        num_classes = 100

    elif args.dataset.lower() == "tinyimagenet_last":
        transform_train = get_pipeline(
            args.normalize, args.train_augment, dataset=args.dataset, device = device
        )
        transform_test = get_pipeline(args.normalize, False, dataset=args.dataset, device = device)
        old_transform_test = get_transform(args.normalize, False, dataset=args.dataset)

        cleanset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            classes="lasthalf"
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            classes="lasthalf"
        )
        dataset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transforms.ToTensor(),
            classes="lasthalf"
        )
        num_classes = 100

    elif args.dataset.lower() == "tinyimagenet_all":
        transform_train = get_pipeline(
            args.normalize, args.train_augment, dataset=args.dataset, device = device
        )
        transform_test = get_pipeline(args.normalize, False, dataset=args.dataset, device = device)
        old_transform_test = get_transform(args.normalize, False, dataset=args.dataset)

        cleanset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            classes="all"
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            classes="all"
        )
        dataset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transforms.ToTensor(),
            classes="all"
        )
        num_classes = 200

    else:
        print("Dataset not yet implemented. Exiting from poison_test.py.")
        sys.exit()

    # Note that trainset has (img, label, indicator)
    trainset = PoisonedDataset(
        cleanset, poison_tuples, args.trainset_size, None, poison_indices
    )

    if simple:
        transform_train: List[Operation] = [SimpleRGBImageDecoder()]#, ToDevice(device, non_blocking=True)]
        transform_test: List[Operation] = [SimpleRGBImageDecoder()]#, ToDevice(device, non_blocking=True)]


    # Convert to FFCV
    print("Writing FFCV Datasets...")
    trainset_ffcv = write_ffcv(args.dataset.lower(), 'train', trainset, args)
    print("Finished writing trainset...")	    
    testset_ffcv = write_ffcv(args.dataset.lower(), 'test', testset, args)
    print("Finished writing testset.")
    # Make loaders
    loaders = {}
    for name in ['train', 'test']:
        BATCH_SIZE = args.batch_size if name == 'train' else 64

        pipelines = {}

        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        indicator_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        
        if name == 'train':
            pipelines['image'] = transform_train
        
        else:
            pipelines['image'] = transform_test
        
        pipelines['label'] = label_pipeline

        if name == 'train':
            pipelines['indicator'] = indicator_pipeline

        loaders[name] = Loader(f"{WRITE_PATH}/{args.dataset.lower()}_{name}_{args.poisons_path.split('/')[1]}.beton",
                                batch_size = BATCH_SIZE,
                                num_workers = 8,
                                order = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL,
                                pipelines = pipelines)
        


    return (
        loaders['train'],
        loaders['test'],
        dataset,
        transform_train,
        transform_test,
        num_classes,
        old_transform_test
    )

def test(net, testloader, device, extra_transforms = None):
    """Function to evaluate the performance of the model
    input:
        net:        Pytorch network object
        testloader: FFCV dataloader object
        device:     Device on which data is to be loaded (cpu or gpu)
    return
        Testing accuracy
    """
    net.eval()
    natural_correct = 0
    total = 0
    results = {}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if extra_transforms:
                inputs = torch.stack([extra_transforms(x) for x in inputs])
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast():
                natural_outputs = net(inputs)
                _, natural_predicted = natural_outputs.max(1)
                natural_correct += natural_predicted.eq(targets).sum().item()

                total += targets.size(0)

                natural_acc = 100.0 * natural_correct / total
                results["Clean acc"] = natural_acc

    return natural_acc


def train(net, trainloader, optimizer, criterion, device, scaler, train_bn=True, extra_transforms = None):
    """Function to perform one epoch of training
    input:
        net:            Pytorch network object
        trainloader:    FFCV dataloader object
        optimizer:      Pytorch optimizer object
        criterion:      Loss function

    output:
        train_loss:     Float, average loss value
        acc:            Float, percentage of training data correctly labeled
    """
    
    # Set net to train and zeros stats
    if train_bn:
        net.train()
    else:
        net.eval()

    net = net.to(device)

    train_loss = 0
    correct = 0
    total = 0
    poisons_correct = 0
    poisons_seen = 0

    for batch_idx, (inputs, targets, p) in enumerate(trainloader):
        if extra_transforms:
            #if batch_idx == 0:
            #   print(type(inputs))
            #   print('here', inputs.shape, targets)
            #   print(extra_transforms)
            inputs = torch.stack([extra_transforms(x) for x in inputs])
        #print('out')
        inputs, targets, p = inputs.to(device), targets.to(device), p.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        poisons_correct += (predicted.eq(targets) * p).sum().item()
        poisons_seen += p.sum().item()

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    return train_loss, acc

