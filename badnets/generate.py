#Generate poisons for Badnets
import pickle
from torchvision import datasets, transforms
import torch

#TODO: allow for general setups later
POISON_SETUPS_PATH = "../poison_setups/cifar10_transfer_learning.pickle"
trainset = trainset = datasets.CIFAR10(root="../data", train=True, download=True,
                                        transform=transforms.ToTensor())
testset = datasets.CIFAR10(root="../data", train=False, download=True,
                                        transform=transforms.ToTensor())


with open(POISON_SETUPS_PATH, "rb") as handle:
    setup_dicts = pickle.load(handle)

def sample_random(dataset, label, n):
    '''
    Randomly sample from dataset for n samples of class label. 
    '''
    
    images = torch.stack([x for x, y in dataset if y == label])
    idxs = torch.randperm(len(images))[:n]
    
    return images[idxs]

def apply_yellow_patch(batch, start_x = 0, start_y = 0, size = 5):
    '''
    Apply a trigger (yellow patch) onto a batch of images.
    '''
    
    patch = torch.stack([torch.ones(size,size), torch.ones(size,size), torch.zeros(size,size)])
    batch[:,:, start_y:start_y + size, start_x: start_x + 5] = patch

    return batch

def generate_poison(setup, trainset, testset):

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
    poisons = apply_patch(clean_bases)

    # format poisons

    # save poisons, labels, and target

    # poison_tuples should be a list of tuples with two entries each (img, label), example:
    # [(poison_0, label_0), (poison_1, label_1), ...]
    # where poison_0, poison_1 etc are PIL images (so they can be loaded like the CIFAR10 data in pytorch)
    with open("poisons.pickle", "wb") as handle:
        pickle.dump(poison_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # base_indices should be a list of indices witin the CIFAR10 data of the bases, this is used for testing for clean-lable
    # i.e. that the poisons are within the l-inf ball of radius 8/255 from their respective bases
    with open("base_indices.pickle", "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # For triggerless attacks use this
    with open("target.pickle", "wb") as handle:
        pickle.dump((transforms.ToPILImage()(target_img), target_label), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # For triggered backdoor attacks use this where patch is a 3x5x5 tensor conataing the patch 
    # and [startx, starty] is the location of the top left pixel of patch in the pathed target 
    with open("target.pickle", "wb") as handle:
        pickle.dump((transforms.ToPILImage()(target_img), target_label, patch, [startx, starty]), handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)

if "__name__" == "__main__":

    print("TESTING POISON GENERATION")
    generate_poison(setup_dicts[0])
    print("FINISHED GENERATING POISONS")