"""nnU-Net 2.8.1 trainer for Dataset901: default nnUNetTrainer plus slab-coverage and thick-slice simulation.

Preregistered in analysis/reports/model_insight_20261012/PREREGISTRATION.md (commit db0521a). Only the training
transforms change; plans, split, epochs, loss, optimiser and checkpoint selection are inherited unchanged.
nnU-Net finds this class through nnUNet_extTrainer=experiments/nnunet_trainers (set by experiments/run.py).

Patch axes are (head-feet, front-back, left-right): Dataset901 inputs are stored LPS and the plans transpose is identity.
"""
import json
from pathlib import Path

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

SLAB_P, THICK_P, FRONT_BACK_SHARE = 0.5, 0.5, 0.15
SLAB_MM, THICK_MM = (30.0, 55.0), (1.0, 3.5)  # MC-RC slab coverage 32.8-52.8 mm, slice spacing 0.91-3.48 mm
# nnunetv2/inference/predict_from_raw_data.py:690-692 pads the z-scored input with 0 when it is smaller than the patch.
PAD_VALUE = 0.0


def thick_slice_operator(n, spacing, target):
    """(n x n) matrix: area-average `spacing`-mm slices into `target`-mm slices, then linear interpolation back."""
    edges = np.arange(n + 1) * spacing
    n_out = int(np.ceil(n * spacing / target - 1e-6))
    lo = np.arange(n_out) * target
    hi = np.minimum(lo + target, edges[-1])
    overlap = np.clip(np.minimum(hi[:, None], edges[None, 1:]) - np.maximum(lo[:, None], edges[None, :-1]), 0, None)
    average = overlap / overlap.sum(axis=1, keepdims=True)
    centres_in, centres_out = (edges[:-1] + edges[1:]) / 2, (lo + hi) / 2
    interpolate = np.stack([np.interp(centres_in, centres_out, column) for column in np.eye(n_out)], axis=1)
    return torch.from_numpy(interpolate @ average).float()


class SlabThickTransform(ImageOnlyTransform):
    """Image-only: optional thick slices along one axis, then optional slab (outside set to PAD_VALUE).

    The slab is placed uniformly among positions that contain every foreground voxel of the patch along its axis;
    it is widened to the tumour extent if the tumour is longer. Labels are never modified.
    """

    def __init__(self, spacing, force=None):
        super().__init__()
        self.spacing = [float(s) for s in spacing]
        self.force = force  # None (random) or one of 'none', 'slab', 'thick', 'slab+thick' (smoke-test samples only)
        self.last = None

    def get_parameters(self, **data_dict):
        mode = self.force
        slab = np.random.uniform() < SLAB_P if mode is None else 'slab' in mode
        thick = np.random.uniform() < THICK_P if mode is None else 'thick' in mode
        axis = 1 if slab and np.random.uniform() < FRONT_BACK_SHARE else 0
        params = dict(axis=axis, slab=None, thick_mm=float(np.random.uniform(*THICK_MM)) if thick else None)
        if slab:
            size = data_dict['image'].shape[1 + axis]
            n = int(round(np.random.uniform(*SLAB_MM) / self.spacing[axis]))
            seg = data_dict['segmentation'][0] > 0
            occupied = torch.nonzero(seg.movedim(axis, 0).reshape(size, -1).any(dim=1))[:, 0]
            if len(occupied):
                first, last = int(occupied.min()), int(occupied.max())
                n = max(n, last - first + 1)
                start = int(np.random.randint(last - n + 1, first + 1))
            else:  # no tumour in this patch: keep at least half of the slab inside it
                start = int(np.random.randint(-(n // 2), size - n + n // 2 + 1))
            params['slab'] = (start, n)
        self.last = dict(params, slab_mm=None if params['slab'] is None else params['slab'][1] * self.spacing[axis])
        return params

    def _apply_to_image(self, img, axis, slab, thick_mm):
        dim = axis + 1
        if thick_mm is not None and thick_mm > self.spacing[axis]:
            operator = thick_slice_operator(img.shape[dim], self.spacing[axis], thick_mm).to(img.dtype)
            img = torch.tensordot(operator, img.movedim(dim, 0), dims=1).movedim(0, dim).contiguous()
        if slab is not None:
            start, n = slab
            keep = torch.zeros(img.shape[dim], dtype=torch.bool)
            keep[max(start, 0):max(start + n, 0)] = True
            shape = [1] * img.ndim
            shape[dim] = -1
            img = torch.where(keep.view(shape), img, torch.tensor(PAD_VALUE, dtype=img.dtype))
        return img


class nnUNetTrainerSlabThick(nnUNetTrainer):
    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None):
        """The default pipeline with SlabThickTransform inserted after the intensity transforms, before mirroring."""
        compose = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        anchors = [i for i, t in enumerate(compose.transforms) if isinstance(t, (MirrorTransform, RemoveLabelTansform))]
        compose.transforms.insert(anchors[0], SlabThickTransform(self.configuration_manager.spacing))
        self.print_to_log_file(f'SlabThickTransform inserted at position {anchors[0]}: slab p={SLAB_P} '
                               f'{SLAB_MM} mm (front-back share {FRONT_BACK_SHARE}), thick p={THICK_P} {THICK_MM} mm, '
                               f'pad value {PAD_VALUE}; labels unchanged')
        return compose


def smoke_test(trainer, sample_dir, samples=(('none', 2), ('slab', 2), ('slab+thick', 2))):
    """Save forced-mode augmented training patches as NIfTI, then train trainer.num_epochs epochs (no validation)."""
    import nibabel as nib
    trainer.initialize()
    loader, _ = trainer.get_dataloaders()
    data_loader = loader.data_loader
    data_loader.oversample_foreground_percent = 1.0  # tumour-containing patches for visual review
    # by name: nnU-Net's nnUNet_extTrainer loader imports this file as a separate module object
    [transform] = [t for t in data_loader.transforms.transforms if type(t).__name__ == 'SlabThickTransform']
    spacing = trainer.configuration_manager.spacing
    affine = np.diag([-spacing[2], -spacing[1], spacing[0], 1.0])  # array (S, P, L) -> NIfTI (L, P, S)
    records = []
    for mode, count in samples:
        transform.force = mode
        for index in range(count):
            batch = next(loader)
            name = f'{len(records) + 1}_{mode.replace("+", "_")}'
            target = batch['target'][0] if isinstance(batch['target'], list) else batch['target']
            directory = Path(sample_dir) / name
            directory.mkdir(parents=True)
            nib.save(nib.Nifti1Image(batch['data'][-1, 0].permute(2, 1, 0).numpy().astype(np.float32), affine),
                     directory / 'image.nii.gz')
            nib.save(nib.Nifti1Image(target[-1, 0].permute(2, 1, 0).numpy().astype(np.uint8), affine),
                     directory / 'gt.nii.gz')
            records.append(dict(sample=name, mode=mode, **{k: v for k, v in transform.last.items()}))
            trainer.print_to_log_file(f'sample {name}: {transform.last}')
    transform.force = None
    (Path(sample_dir) / 'samples.json').write_text(json.dumps(records, indent=2, default=str) + '\n')
    trainer.run_training()
