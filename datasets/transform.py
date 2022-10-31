from torchvision import transforms

mean, std = {}, {}
mean["cifar10"] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean["cifar100"] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean["svhn"] = [0.4380, 0.4440, 0.4730]
mean["stl10"] = [x / 255 for x in [112.4, 109.1, 98.6]]

std["cifar10"] = [x / 255 for x in [63.0, 62.1, 66.7]]
std["cifar100"] = [x / 255 for x in [68.2, 65.4, 70.4]]
std["svhn"] = [0.1751, 0.1771, 0.1744]
std["stl10"] = [x / 255 for x in [68.4, 66.6, 68.5]]


def get_transform_by_name(dset_name="cifar10", train=False):
    dset_name = dset_name.lower()
    crop_size = 96 if dset_name == "stl10" else 32
    transform = get_transform(mean[dset_name], std[dset_name], crop_size, train)
    return transform


def get_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
