"""Copy the Fable 5.2 high masks into the MMIV reviewer bundle (cohort "Model insight · MC-RC").

  python3 harness/fable/fable_export.py

Writes ~/MMIV/data/vs_viewer_bundle/data/mcrc/<patient_id>/pred_llm_fable52_high_run1.nii.gz
for all 22 patients, on the reference label's grid, in the same way as the Opus and Astra
exports. A case without a valid saved mask is exported as an empty mask.
"""
import csv
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("original_score", ROOT / "harness/score.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
ARM = ROOT / "runs/fable_high"
NAME = "pred_llm_fable52_high_run1.nii.gz"


def main():
    records = list(csv.DictReader((ROOT / "private/case_key.csv").open()))
    assert len(records) == 22
    exported = {}
    for r in sorted(records, key=lambda r: r["patient_id"]):
        pid = r["patient_id"]
        ref = nib.load(original.DATA / pid / "label.nii.gz")
        status = json.loads((ARM / pid / "status.json").read_text())["status"]
        mask = (original.load_mask(ARM / pid / "label.nii.gz", ref) if status == "ok"
                else np.zeros(ref.shape, bool))
        im = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header)
        im.set_data_dtype(np.uint8)
        dest = original.BUNDLE / pid / NAME
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".fable-", suffix=".nii.gz", delete=False) as f:
            temp = Path(f.name)
        try:
            nib.save(im, temp)
            os.replace(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
        os.chmod(dest, 0o644)
        exported[pid] = str(dest)
        print(f"EXPORTED {pid}: {int(mask.sum())} voxels", flush=True)
    (ROOT / "results/fable_high/reviewer_exports.json").write_text(json.dumps(exported, indent=2) + "\n")


if __name__ == "__main__":
    main()
