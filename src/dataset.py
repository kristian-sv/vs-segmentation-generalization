"""Prepare MRI volumes, then sample training patches. Validation uses whole volumes."""
from monai.data import CacheDataset, Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, NormalizeIntensityd, Lambdad, DivisiblePadd, SpatialPadd,
    RandCropByPosNegLabeld, RandGaussianNoised, RandAdjustContrastd,
    RandShiftIntensityd, RandFlipd,
)
from data import development_pairs


def base_transforms(settings):
    """Same deterministic image preparation for training, validation and export."""
    keys = ['image', 'label']
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes='RAS'),
        Spacingd(keys=keys, pixdim=settings['pixdim'], mode=('bilinear', 'nearest')),
        CropForegroundd(keys=keys, source_key='image', margin=5),
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        Lambdad(keys=['label'], func=lambda x: (x > 0).float()),
        DivisiblePadd(keys=keys, k=16, mode='edge'),
        SpatialPadd(keys=keys, spatial_size=settings['patch_size'], mode='edge'),
    ])


def training_transforms(settings):
    """The existing baseline augmentation; only dropout varies between our models."""
    keys = ['image', 'label']
    return Compose([
        RandCropByPosNegLabeld(keys=keys, label_key='label', spatial_size=settings['patch_size'],
                              pos=settings['pos_neg_ratio'], neg=1,
                              num_samples=settings['num_samples'], allow_smaller=True),
        RandGaussianNoised(keys=['image'], prob=0.3, std=0.1),
        RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.7, 1.5)),
        RandShiftIntensityd(keys=['image'], prob=0.3, offsets=0.1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
    ])


def get_loader(lock, split, settings):
    pairs = development_pairs(lock, split)  # rejects test/external partitions
    cached = CacheDataset(pairs, base_transforms(settings), cache_rate=1,
                          num_workers=settings['num_workers'])
    dataset = Dataset(cached, training_transforms(settings)) if split == 'train' else cached
    return DataLoader(dataset, batch_size=settings['batch_size'] if split == 'train' else 1,
                      shuffle=split == 'train', num_workers=settings['num_workers'], pin_memory=True)
