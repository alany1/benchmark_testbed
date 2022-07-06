import csv
import datetime
import os
import sys

import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from models import *
from tinyimagenet_module import TinyImageNet
from learning_module import TINYIMAGENET_ROOT, get_transform, PoisonedDataset

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

WRITE_PATH = './ffcv_files'

def write_ffcv(name, set_type, dataset):
    """
    Write {name} (cifar10, tinyimagenet_first, etc.) dataset of type set_type (train or test) to FFCV dataset. 
    """
    writer = DatasetWriter(f"{WRITE_PATH}/{name}_{set_type}.beton", {'image': RGBImageField(), 'label': IntField()})
    writer.from_indexed_dataset(dataset))

def get_dataset(args, poison_tuples, poison_indices):
    if args.dataset.lower() == "cifar10":
        transform_train = get_transform(args.normalize, args.train_augment)
        transform_test = get_transform(args.normalize, False)
        cleanset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        num_classes = 10

    elif args.dataset.lower() == "tinyimagenet_first":
        transform_train = get_transform(
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
        cleanset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_train,
            classes="firsthalf",
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="firsthalf",
        )
        dataset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transforms.ToTensor(),
            classes="firsthalf",
        )
        num_classes = 100

    elif args.dataset.lower() == "tinyimagenet_last":
        transform_train = get_transform(
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
        cleanset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_train,
            classes="lasthalf",
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="lasthalf",
        )
        dataset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transforms.ToTensor(),
            classes="lasthalf",
        )
        num_classes = 100

    elif args.dataset.lower() == "tinyimagenet_all":
        transform_train = get_transform(
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
        cleanset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_train,
            classes="all",
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="all",
        )
        dataset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transforms.ToTensor(),
            classes="all",
        )
        num_classes = 200

    else:
        print("Dataset not yet implemented. Exiting from poison_test.py.")
        sys.exit()

    trainset = PoisonedDataset(
        cleanset, poison_tuples, args.trainset_size, transform_train, poison_indices
    )

    # Convert to FFCV
    trainset_ffcv = write_ffcv(args.dataset.lower(), 'train', trainset)
    testset_ffcv = write_ffcv(args.dataset.lower(), 'test', testset)
    
    # Make loaders
    loaders = {}
    for name in ['train', 'test']:
        BATCH_SIZE = 64 if name == 'test' else args.batch_size
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        loaders[name] = Loader(f'{WRITE_PATH}/{args.dataset.lower()}_{name}.beton',
                            batch_size=BATCH_SIZE,
                            num_workers=8,
                            order=OrderOption.RANDOM,
                            drop_last=(name == 'train'),
                            pipelines={'image': image_pipeline,
                                       'label': label_pipeline})


    return (
        trainloader,
        testloader,
        dataset,
        transform_train,
        transform_test,
        num_classes,
    )
