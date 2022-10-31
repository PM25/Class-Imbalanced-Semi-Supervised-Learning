# import needed library
import json
import random
import logging
import warnings
import numpy as np
from pathlib import Path
from labelshift.lse import LSE

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from utils.utils import over_write_args_from_file
from datasets.transform import get_transform_by_name
from datasets.cissl_dataset import CISSL_Dataset


def main(args):
    if args.seed is not None:
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely disable data parallelism.")
    print()

    # Construct Dataset & DataLoader
    train_dset = CISSL_Dataset(alg="shiftestimator", name=args.dataset, train=True, num_classes=args.num_classes, data_dir=args.data_dir)
    lb, ulb = train_dset.get_lb_ulb_data(
        args.max_labeled_per_class,
        args.max_unlabeled_per_class,
        args.lb_imb_ratio,
        args.ulb_imb_ratio,
        args.imb_type,
        dset=False,
        include_lb_to_ulb=False,
        seed=args.seed,
    )

    lb_data, lb_targets = lb
    ulb_data, _ = ulb

    train_transform = get_transform_by_name(args.dataset, train=True)
    test_transform = get_transform_by_name(args.dataset, train=False)

    estimator = LSE(args=args, save_path=args.save_model_path)
    estimator.train_base_models(lb_data, lb_targets, args.num_classes, train_transform=train_transform, test_transform=test_transform)
    estimations = estimator.estimate(ulb_data, test_transform=test_transform, verbose=True)

    for name, estimation in estimations.items():
        save_file_path = Path(args.save_estimation_path) / f"{name}_estimation.json"
        save_file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_file_path, "w") as f:
            json.dump({"distribution": estimation["estimation"]}, f)

    logging.warning("\nAll estimations saved successfully")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")

    """
    Save Configuration
    """
    parser.add_argument("--save_model_path", type=str, default="./saved_models/label_shift")
    parser.add_argument("--save_estimation_path", type=str, default="./dist_estimation")
    parser.add_argument("--save_model", type=str2bool, default=True, help="whether to save model")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="whether to overwrite exist models")

    """
    Training Configuration
    """
    parser.add_argument("--num_train_iter", type=int, default=16000, help="total number of training iterations")
    parser.add_argument("--num_eval_iter", type=int, default=1600, help="evaluation frequency")
    parser.add_argument("--batch_size", type=int, default=64, help="total number of batch size of labeled data")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="batch size of evaluation data loader (it does not affect the accuracy)",
    )
    parser.add_argument("--ema_m", type=float, default=0.999)
    parser.add_argument("--use_mixup_drw", type=str2bool, default=True, help="whether to use mixup + deferred reweighting")
    parser.add_argument(
        "--drw_warm", type=float, default=0.75, help="deferred reweighting warm iterations out of total training iterations"
    )

    """
    Optimizer Configurations
    """
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--bn_momentum", type=float, default=0.001)
    parser.add_argument("--amp", type=str2bool, default=False, help="use mixed precision training or not")
    parser.add_argument("--clip", type=float, default=0)

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="WideResNet")
    parser.add_argument("--net_from_name", type=str2bool, default=False)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    """
    Data Configurations
    """
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_labeled_per_class", type=int, default=1500, help="the maximum number of labeled data per class")
    parser.add_argument("--max_unlabeled_per_class", type=int, default=3000, help="the maximum number of unlabeled data per class")
    parser.add_argument("--num_val_per_class", type=int, default=5, help="number of validations per class")

    """
    Imbalanced Configurations
    """
    parser.add_argument("--imb_type", type=str, default="long", help="long tailed or step imbalanced")
    parser.add_argument("--lb_imb_ratio", type=float, default=1, help="imbalance ratio of labeled data")
    parser.add_argument("--ulb_imb_ratio", type=float, default=1, help="imbalance ratio of unlabeled data")

    """
    Label Shift Estimation Configurations
    """
    parser.add_argument("--lse_algs", type=json.loads, default=["MLLS"], help="list of label shift estimation methods to use")
    parser.add_argument("--calibrations", type=json.loads, default=["BCTS"], help="list of calibration methods to use")
    parser.add_argument("--num_ensemble", type=int, default=10, help="number of subsets for training ensemble models")

    """
    GPU Configurations
    """
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    # config file
    parser.add_argument("--c", type=str, default="")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    main(args)
