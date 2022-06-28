#Generate poisons for Badnets
import pickle
from torchvision import datasets, transforms
import torch
import argparse
import sys
sys.path.insert(1, '/Users/alanyu/Desktop/poisoning-benchmark/')
from learning_module import TinyImageNet, TINYIMAGENET_ROOT

def sample_random(dataset, label, n):
    '''
    Randomly sample from dataset for n samples of class label. 
    '''
    
    images = torch.stack([x for x, y in dataset if y == label])
    idxs = torch.randperm(len(images))[:n]
    
    return images[idxs]

def apply_patch(batch, patch, start_x = 0, start_y = 0, size = 5):
    '''
    Apply an arbitrary patch onto a batch of images.
    '''
    
    batch[:,:, start_y:start_y + size, start_x: start_x + size] = patch
    return batch

def get_yellow_patch(size = 5):
    '''
    Apply a trigger (yellow patch) onto a batch of images.
    '''
    
    return torch.stack([torch.ones(size,size), torch.ones(size,size), torch.zeros(size,size)])
    
def generate_poison(directory, setup, trainset, testset, patch, start_x = 0, start_y = 0, size = 5):

    target_class = setup["target class"]
    target_img_idx = setup["target index"]
    poisoned_label = setup["base class"]
    base_indices = setup["base indices"]
    num_poisons = len(base_indices)
    
    target_img, target_label = testset[target_img_idx]

    #note that for Badnets, we don't care about the base images
    base_imgs = torch.stack([trainset[i][0] for i in base_indices])
    base_labels = torch.LongTensor([trainset[i][1] for i in base_indices])

    # sample random clean imgs from target class
    clean_bases = sample_random(trainset, target_label, len(base_labels))
    # slap a patch on
    poisons = apply_patch(clean_bases, patch, start_x, start_y, size)

    # format poisons
    t = transforms.ToPILImage()
    #print(base_labels[i])
    poison_tuples = [(t(poisons[i]), base_labels[i].item()) for i in range(len(poisons))]

    # save poisons, labels, and target

    # poison_tuples should be a list of tuples with two entries each (img, label), example:
    # [(poison_0, label_0), (poison_1, label_1), ...]
    # where poison_0, poison_1 etc are PIL images (so they can be loaded like the CIFAR10 data in pytorch)
    with open(f"{directory}/poisons.pickle", "wb") as handle:
        pickle.dump(poison_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # base_indices should be a list of indices within the CIFAR10 data of the bases, this is used for testing for clean-label
    # i.e. that the poisons are within the l-inf ball of radius 8/255 from their respective bases
    # !!!For Badnets, this step is not important. The trigger is obvious so the image is not like the target, and the 
    # images are definitely not like the original base images.!!!
    with open(f"{directory}/base_indices.pickle", "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # For triggered backdoor attacks use this where patch is a 3x5x5 tensor conataing the patch 
    # and [startx, starty] is the location of the top left pixel of patch in the pathed target 
    with open(f"{directory}/target.pickle", "wb") as handle:
        pickle.dump((transforms.ToPILImage()(target_img), target_label, patch, [start_x, start_y]), handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Construct Badnets poisons")
    parser.add_argument('-t', '--trials', type = int, metavar='', required=True, help='Number of trials to sample')
    parser.add_argument('-d', '--directory', type = str, metavar = '', required = True, help = 'Directory to store poison pickles')
    parser.add_argument('-s', '--dataset', type = str, metavar = '', required = True, help = 'Dataset (cifar or imagenet')
    parser.add_argument('-m', '--method', type = str, metavar = '', required = True, help = 'Training method (transfer or scratch')
    
    args = parser.parse_args()
    
    size = 5
    POISONS_SETUP_PATH = None
    trainset, testset = None, None
    if args.dataset == 'cifar':
        if args.method == 'transfer':
            POISON_SETUPS_PATH = "/Users/alanyu/Desktop/poisoning-benchmark/poison_setups/cifar10_transfer_learning.pickle"
        elif args.method == 'scratch':
            POISON_SETUPS_PATH = "/Users/alanyu/Desktop/poisoning-benchmark/poison_setups/cifar10_from_scratch.pickle"
        else:
            print("Invalid Method. Exiting...")
            sys.exit(0)
        trainset = datasets.CIFAR10(root="/Users/alanyu/Desktop/poisoning-benchmark/data", train=True, download=True,
                                        transform=transforms.ToTensor())
        testset = datasets.CIFAR10(root="/Users/alanyu/Desktop/poisoning-benchmark/data", train=False, download=True,
                                        transform=transforms.ToTensor())
    elif args.dataset == 'imagenet':
        size = 8
        if args.method == 'transfer':
            #Use the latter half of the tinyimagenet dataset
            POISON_SETUPS_PATH = "/Users/alanyu/Desktop/poisoning-benchmark/poison_setups/tinyimagenet_transfer_learning.pickle"

            trainset = TinyImageNet(
                TINYIMAGENET_ROOT,
                split="train",
                transform=transforms.ToTensor(),
                classes="lasthalf"
            )

            testset = TinyImageNet(
                TINYIMAGENET_ROOT,
                split="val",
                transform=transforms.ToTensor(),
                classes="lasthalf"
            )

        elif args.method == 'scratch':
            trainset = TinyImageNet(
                TINYIMAGENET_ROOT,
                split="train",
                transform=transforms.ToTensor(),
                classes="all"
            )

            testset = TinyImageNet(
                TINYIMAGENET_ROOT,
                split="test",
                transform=transforms.ToTensor(),
                classes="all"
            )
            POISON_SETUPS_PATH = "/Users/alanyu/Desktop/poisoning-benchmark/poison_setups/tinyimagenet_from_scratch.pickle"

        else:
            print("Invalid Method. Exiting...")
            sys.exit(0)   
    else:
        print('Dataset not supported yet. Exiting...')
        sys.exit(0)
    
    with open(POISON_SETUPS_PATH, "rb") as handle:
        setup_dicts = pickle.load(handle)

    YELLOW = get_yellow_patch(size)

    print("--------Starting Poison Generation-------")


    for i in range(args.trials):
        generate_poison(f"{args.directory}/badnets_poisons/{i}", setup_dicts[i], trainset, testset, YELLOW, size = size)
        if i%10==0:
            print('Finished Trial', i+1 )


    print("--------Finished Generating Poisons--------")