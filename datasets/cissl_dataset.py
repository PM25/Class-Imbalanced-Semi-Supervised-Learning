import json
import numpy as np
from pathlib import Path

import torchvision

from .dataset import BasicDataset
from .data_utils import split_cissl_data, sample_imb_data
from .transform import get_transform_by_name


class CISSL_Dataset:
    """
    CISSL_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self, alg, name="cifar10", train=True, num_classes=10, data_dir="./data"):
        """
        Args
            alg: CISSL algorithms
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        self.alg = alg
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.transform = get_transform_by_name(self.name, train)

    def get_data(self, svhn_extra=True):
        """
        get_data returns data (images) and targets (labels)
        shape of data: B, H, W, C
        shape of labels: B,
        """
        dset = getattr(torchvision.datasets, self.name.upper())
        if "CIFAR" in self.name.upper():
            dset = dset(self.data_dir, train=self.train, download=True)
            data, targets = dset.data, dset.targets
            return data, targets

        elif self.name.upper() == "SVHN":
            if self.train:
                if svhn_extra:  # train+extra
                    dset_base = dset(self.data_dir, split="train", download=True)
                    data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
                    dset_extra = dset(self.data_dir, split="extra", download=True)
                    data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
                    data = np.concatenate([data_b, data_e])
                    targets = np.concatenate([targets_b, targets_e])
                    del data_b, data_e
                    del targets_b, targets_e
                else:  # train_only
                    dset = dset(self.data_dir, split="train", download=True)
                    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            else:  # test
                dset = dset(self.data_dir, split="test", download=True)
                data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            return data, targets

        elif self.name.upper() == "STL10":
            split = "train" if self.train else "test"
            dset_lb = dset(self.data_dir, split=split, download=True)
            dset_ulb = dset(self.data_dir, split="unlabeled", download=True)
            data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
            ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])
            return data, targets, ulb_data

    def get_dset(self, is_ulb=False, strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.

        Args
            is_ulb: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True。
            onehot: If True, the label is not integer, but one-hot vector.
        """
        if self.name.upper() == "STL10":
            data, targets, _ = self.get_data()
        else:
            data, targets = self.get_data()

        return BasicDataset(self.alg, data, targets, self.num_classes, self.transform, is_ulb, strong_transform, onehot)

    def get_lb_ulb_data(
        self,
        max_labeled_per_class,
        max_unlabeled_per_class,
        lb_imb_ratio,
        ulb_imb_ratio,
        imb_type,
        include_lb_to_ulb=True,
        strong_transform=None,
        onehot=False,
        dset=True,
        seed=0,
    ):
        """
        get_cissl_dset split training samples into labeled and unlabeled samples.
        The labeled and unlabeled data might be imbalanced over classes.

        Args:
            num_labels: number of labeled data.
            lb_img_ratio: imbalance ratio of labeled data.
            ulb_imb_ratio: imbalance ratio of unlabeled data.
            imb_type: type of imbalance data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            dset: If True, return BasicDataset else return raw data.
            seed: Get deterministic results of labeled and unlabeld data.

        Returns:
        if dset = True:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        else:
            labeled (data, targets), unlabeled (data, None)
        """
        # Supervised top line using all data as labeled data.
        if self.alg == "fullysupervised":
            lb_data, lb_targets = self.get_data()
            lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes, self.transform, False, None, onehot)
            if dset:
                return lb_dset, None
            else:
                return (lb_data, lb_targets), None

        if self.name.upper() == "STL10":
            lb_data, lb_targets, ulb_data = self.get_data()
            if include_lb_to_ulb:
                ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
            lb_data, lb_targets, _, lb_class_num_list = sample_imb_data(
                lb_data, lb_targets, max_labeled_per_class, lb_imb_ratio, imb_type, self.num_classes, seed=seed
            )
            ulb_targets = None
            print(f"#labeled  : {lb_class_num_list.sum()}, {lb_class_num_list}")
            print(f"#unlabeled: unknown")

        else:
            data, targets = self.get_data()
            lb_data, lb_targets, ulb_data, ulb_targets = split_cissl_data(
                data,
                targets,
                max_labeled_per_class,
                max_unlabeled_per_class,
                lb_imb_ratio,
                ulb_imb_ratio,
                imb_type,
                self.num_classes,
                include_lb_to_ulb,
                seed=seed,
            )

        # output the distribution of labeled data for remixmatch
        count = np.zeros(self.num_classes)
        for c in lb_targets:
            count[c] += 1
        dist = count / count.sum()
        out = {"distribution": dist.tolist()}
        output_folder = Path("./data_statistics/")
        output_folder.mkdir(exist_ok=True, parents=True)
        output_path = output_folder / f"{self.name}_n{max_labeled_per_class}_m{max_unlabeled_per_class}_l{lb_imb_ratio}_{imb_type}.json"
        with open(output_path, "w") as w:
            json.dump(out, w)

        if dset:
            lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes, self.transform, False, None, onehot)
            ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets, self.num_classes, self.transform, True, strong_transform, onehot)
            return lb_dset, ulb_dset
        else:
            return (lb_data, lb_targets), (ulb_data, None)
