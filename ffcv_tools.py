import csv
import datetime
import os
import sys
import pip

import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from models import *
from tinyimagenet_module import TinyImageNet
from learning_module import TINYIMAGENET_ROOT, PoisonedDataset

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

WRITE_PATH = './ffcv_files'

data_mean_std_dict = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet_all": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    "tinyimagenet_first": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    "tinyimagenet_last": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
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

def write_ffcv(name, set_type, dataset):
    """
    Write {name} (cifar10, tinyimagenet_first, etc.) dataset of type set_type (train or test) to FFCV dataset. 
    """
    writer = DatasetWriter(f"{WRITE_PATH}/{name}_{set_type}.beton", {'image': RGBImageField(), 'label': IntField()})
    writer.from_indexed_dataset(dataset))

def get_pipeline(normalize, augment, dataset="CIFAR10"):
    """Function to perform required transformation on the tensor
    input:
        normalize:      Bool value to determine whether to normalize data
        augment:        Bool value to determine whether to augment data
        dataset:        Name of the dataset
    return
        Pytorch tranforms.Compose with list of all transformations
    """

    dataset = dataset.lower()
    mean, std = data_mean_std_dict[dataset]
    if "tinyimagenet" in dataset:
        dataset = "tinyimagenet"
    cropsize = {"cifar10": 32, "cifar100": 32, "tinyimagenet": 64}[dataset]
    padding = 4

    pipeline: List[Operation] = [SimpleRGBDecoder()]

    if normalize and augment:
        transform_list = [
            transforms.RandomCrop(cropsize, padding = padding), 
            RandomHorizontalFlip(), 
            ToTensor(),
            ToDevice('cuda', non_blocking = True),
            ToTorchImage(),
            transforms.normalize(mean, std)
        ]
    elif augment:
        transform_list = [
            transforms.RandomCrop(cropsize, padding = padding),
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice('cuda', non_blocking = True),
            ToTorchImage()
        ]
    elif normalize:
        transform_list = [ToTensor(), ToDevice('cuda', non_blocking = True), ToTorchImage(),transforms.Normalize(mean, std)]
    else:
        transform_list = [ToTensor(), ToDevice('cuda', non_blocking = True), ToTorchImage(),]

    pipeline.extend(transform_list)

    return pipeline

def get_dataset(args, poison_tuples, poison_indices):
    if args.dataset.lower() == "cifar10":
        transform_train = get_pipeline(args.normalize, args.train_augment)
        transform_test = get_pipeline(args.normalize, False)
        
        cleanset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True)
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True)
        
        dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.ToTensor())
        
        num_classes = 10

    elif args.dataset.lower() == "tinyimagenet_first":
        transform_train = get_pipeline(
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_pipeline(args.normalize, False, dataset=args.dataset)
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
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_pipeline(args.normalize, False, dataset=args.dataset)
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
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_pipeline(args.normalize, False, dataset=args.dataset)
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

    trainset = PoisonedDataset(
        cleanset, poison_tuples, args.trainset_size, poison_indices
    )

    # Convert to FFCV
    trainset_ffcv = write_ffcv(args.dataset.lower(), 'train', trainset)
    testset_ffcv = write_ffcv(args.dataset.lower(), 'test', testset)
    
    # Make loaders
    loaders = {}
    for name in ['train', 'test']:
        BATCH_SIZE = args.batch_size if name == 'train' else 64

        #image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda'), Squeeze()]
        indicator_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda'), Squeeze()]
        
        if name == 'train':
            image_pipeline = transform_train
        else:
            image_pipeline = transform_test
        
        loaders[name] = Loader(f"ffcv_files/{args.dataset.lower}_{name}.beton",
                                batch_size = BATCH_SIZE
                                num_workers = 8,
                                order = OrderOption.RANDOM,
                                pipelines = {'image': image_pipeline,
                                            'label': label_pipeline,
                                            'indicator': indicator_pipeline})
        


    return (
        loaders['train'],
        loaders['test'],
        dataset,
        transform_train,
        transform_test,
        num_classes,
    )
