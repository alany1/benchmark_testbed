############################################################
#
# benchmark_test.py
# Code to execute benchmark tests
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################
import argparse
import os

import poison_test
import poison_test_ffcv_parallel
from learning_module_parallel import now, model_paths, set_defaults

def test(args):
    if args.ffcv:
        poison_test_ffcv_parallel.main(args)
    else:
        poison_test.main(args)

def main(args):
    """Main function to run a benchmark test
    input:
        args:       Argparse object that contains all the parsed values
    return:
        void
    """

    print(now(), "benchmark_test_parallel.py running.")
    out_dir = args.output
    if not args.from_scratch:
        print(
            f"Testing poisons from {args.poisons_path}, in the transfer learning "
            f"setting...\n"
        )

        models = {
            "cifar10": ["resnet18", "VGG11", "MobileNet_V2"],
            "tinyimagenet_last": ["vgg16", "resnet34", "mobilenet_v2"],
        }[args.dataset.lower()]

        ####################################################
        #          Transfer learning
        print("Transfer learning test:")
        print(args)

        # white-box attack
        print("---------Starting White Box Attack----------")
        args.output = os.path.join(out_dir, "ffe-wb")
        args.model = models[0]
        args.model_path = model_paths[args.dataset]["whitebox"]
        test(args)

        # black box attacks
        print("----------Starting Black Box Attacks----------")
        args.output = os.path.join(out_dir, "ffe-bb")

        args.model = models[1]
        args.model_path = model_paths[args.dataset]["blackbox"][0]
        test(args)

        args.model_path = model_paths[args.dataset]["blackbox"][1]
        args.model = models[2]
        test(args)

    else:
        print(
            f"Testing poisons from {args.poisons_path}, in the from scratch training "
            f"setting...\n"
        )

        ####################################################
        #           From Scratch Training (fst)
        args.model_path = None
        args.output = os.path.join(out_dir, "fst")

        if args.dataset.lower() == "cifar10":
            print(f"From Scratch testing for {args.dataset}")
            args.model = "resnet18"
            test(args)

            # args.model = "MobileNet_V2"
            # test(args)

            # args.model = "VGG11"
            # test(args)

        else:
            print(f"From Scratch testing for {args.dataset}")
            args.model = "vgg16"
            test(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")
    parser.add_argument(
        "--from_scratch", action="store_true", help="Train from scratch with poisons?"
    )
    parser.add_argument(
        "--poisons_path", type=str, required=True, help="where are the poisons?"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset")
    parser.add_argument(
        "--output", default="output_default", type=str, help="output subdirectory"
    )
    parser.add_argument(
        '--ffcv', type=int, required = True, help = 'Train with FFCV? 0/1 for no/yes')

    args = parser.parse_args()

    args.ffcv = bool(args.ffcv)

    set_defaults(args)
    main(args)
