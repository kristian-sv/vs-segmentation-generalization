"""Segmentation tools for the LLM agent, served over MCP (stdio) inside the sandbox.

The image at /data/image.nii.gz is re-stacked to canonical RAS order on load
(axis permutation and flips only, no resampling). Voxel indices (i, j, k) refer
to that canonical array: i increases toward patient Right, j toward Anterior,
k toward Superior. `save` maps the mask back to the original grid exactly.

Every call is logged to /out/tool_log.jsonl and every rendered image to /out/views/.
"""
import functools
import inspect
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mcp.server.fastmcp import FastMCP, Image
from nibabel import orientations as ornt
from scipy import ndimage as ndi

DATA = Path("/data/image.nii.gz")
OUT = Path("/out")
MAX_CALLS = 40
MAX_IMAGES = 20
PANEL_PX = 3.0  # inches per montage panel

orig = nib.load(DATA)
orig_ornt = ornt.io_orientation(orig.affine)
img = nib.as_closest_canonical(orig)
vol = np.asarray(img.dataobj).astype(np.float32)
sp = np.array(img.header.get_zooms()[:3], dtype=float)
VOX_ML = float(np.prod(sp)) / 1000.0
NZ = vol[vol > 0]
WINDOW = tuple(np.percentile(NZ, [0.5, 99.5])) if NZ.size else (0.0, 1.0)
# acquisition slice direction = original array axis 2, expressed as a canonical axis
ACQ_AXIS = int(orig_ornt[2, 0])
AXIS_NAMES = ["Right-Left (i)", "Anterior-Posterior (j)", "Superior-Inferior (k)"]
PLANE_NORMAL = {"sagittal": 0, "coronal": 1, "axial": 2}
NORMAL_PLANE = {v: k for k, v in PLANE_NORMAL.items()}

masks: dict[str, np.ndarray] = {}
state = {"calls": 0, "images": 0, "saved": False}
(OUT / "views").mkdir(parents=True, exist_ok=True)
mcp = FastMCP("seg")


# ---------- helpers ----------

def log(tool, args, result):
    with (OUT / "tool_log.jsonl").open("a") as f:
        f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "n": state["calls"],
                            "tool": tool, "args": args, "result": result}) + "\n")


def budget(tool, args, image=False):
    """Count a call; return an error string if the budget is exhausted."""
    state["calls"] += 1
    if state["calls"] > MAX_CALLS:
        msg = f"Budget exhausted ({MAX_CALLS} calls). Only `save` is still allowed."
    elif image and state["images"] >= MAX_IMAGES:
        msg = f"Image budget exhausted ({MAX_IMAGES} images). Use text tools or `save`."
    else:
        if image:
            state["images"] += 1
        return None
    log(tool, args, msg)
    return msg


def remaining():
    return f"[budget: {MAX_CALLS - state['calls']} calls, {MAX_IMAGES - state['images']} images left]"


def get_mask(name):
    if name not in masks:
        raise ValueError(f"No mask named '{name}'. Existing: {sorted(masks) or 'none'}")
    return masks[name]


def box_slices(box):
    if not box:
        return tuple(slice(0, s) for s in vol.shape)
    if len(box) != 6:
        raise ValueError("box must be [i0, i1, j0, j1, k0, k1] (end exclusive)")
    return tuple(slice(max(0, int(box[2 * a])), min(vol.shape[a], int(box[2 * a + 1]))) for a in range(3))


def describe(m):
    """Compact text summary of a mask, including an area profile along the acquisition axis."""
    n = int(m.sum())
    if n == 0:
        return {"voxels": 0, "volume_ml": 0.0}
    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0)
    lab, ncomp = ndi.label(m, structure=ndi.generate_binary_structure(3, 3))
    sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1]
    a = ACQ_AXIS
    areas = m.sum(axis=tuple(x for x in range(3) if x != a))
    nzs = np.nonzero(areas)[0]
    prof = {int(s): int(areas[s]) for s in nzs}
    if len(prof) > 40:  # bin long profiles so the text stays short
        step = int(np.ceil(len(nzs) / 40))
        prof = {f"{int(nzs[q])}-{int(nzs[min(q + step, len(nzs)) - 1])}": int(areas[nzs[q:q + step]].sum())
                for q in range(0, len(nzs), step)}
    return {
        "voxels": n,
        "volume_ml": round(n * VOX_ML, 3),
        "bbox_ijk": [int(v) for v in np.r_[lo, hi]],
        "extent_mm": [round(float((hi[x] - lo[x] + 1) * sp[x]), 1) for x in range(3)],
        "centroid_ijk": [round(float(v), 1) for v in idx.mean(0)],
        "components": int(ncomp),
        "largest_component_voxels": [int(s) for s in sizes[:5]],
        f"area_per_slice_along_{'ijk'[a]}": prof,
    }


def ball(radius_mm):
    r = [int(round(radius_mm / s)) for s in sp]
    if max(r) == 0:
        return np.ones((1, 1, 1), bool)
    g = np.ogrid[tuple(slice(-x, x + 1) for x in r)]
    d = sum((gi * s / radius_mm) ** 2 for gi, s in zip(g, sp))
    return d <= 1.0


def plane_slice(arr, normal, idx):
    """2D slice displayed with rows = vertical axis, cols = horizontal axis (voxel indices)."""
    if normal == 2:   # axial: horizontal i, vertical j
        return arr[:, :, idx].T, (0, 1)
    if normal == 1:   # coronal: horizontal i, vertical k
        return arr[:, idx, :].T, (0, 2)
    return arr[idx, :, :].T, (1, 2)  # sagittal: horizontal j, vertical k


def render(normal, indices, crop=None, mask=None, title=""):
    """Montage of slices. crop = (h0, h1, v0, v1) in voxel indices of the displayed axes."""
    n = len(indices)
    cols = min(n, 4) if n <= 8 else min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(PANEL_PX * cols, PANEL_PX * rows + 0.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, indices):
        sl, (h, v) = plane_slice(vol, normal, idx)
        ax.imshow(sl, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1], origin="lower", interpolation="nearest")
        if mask is not None:
            ms, _ = plane_slice(mask.astype(float), normal, idx)
            if ms.any():
                ax.contour(ms, levels=[0.5], colors="orange", linewidths=0.9, origin="lower")
        ax.set_aspect(sp[v] / sp[h])
        h0, h1, v0, v1 = crop if crop else (0, sl.shape[1] - 1, 0, sl.shape[0] - 1)
        # radiological display: patient Right on image left (i and j axes run reversed)
        ax.set_xlim(h1, h0) if h in (0, 1) else ax.set_xlim(h0, h1)
        ax.set_ylim(v0, v1)
        ax.axis("on")
        ax.tick_params(labelsize=6, length=2)
        ax.grid(color="cyan", alpha=0.2, linewidth=0.4)
        ax.set_title(f"{NORMAL_PLANE[normal]} {'ijk'[normal]}={idx}", fontsize=8)
        left, right = ("R", "L") if h == 0 else ("A", "P")
        down, up = ("P", "A") if v == 1 else ("I", "S")
        for txt, xy in ((left, (0.01, 0.5)), (right, (0.95, 0.5)), (up, (0.48, 0.92)), (down, (0.48, 0.02))):
            ax.text(*xy, txt, transform=ax.transAxes, color="yellow", fontsize=7)
        ax.set_xlabel("ijk"[h], fontsize=7, labelpad=1)
        ax.set_ylabel("ijk"[v], fontsize=7, labelpad=1)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    png = buf.getvalue()
    (OUT / "views" / f"{state['calls']:02d}_{NORMAL_PLANE[normal]}.png").write_bytes(png)
    return Image(data=png, format="png")


def register(fn, wrapper):
    """Expose `wrapper` with fn's name, docstring and parameters (but no return schema)."""
    functools.update_wrapper(wrapper, fn)
    wrapper.__annotations__ = {k: v for k, v in fn.__annotations__.items() if k != "return"}
    sig = inspect.signature(fn)
    wrapper.__signature__ = sig.replace(return_annotation=inspect.Signature.empty)
    return mcp.tool()(wrapper)


def tool(fn):
    """Wrap a text tool: budget, error handling, logging."""
    def wrapper(**kw):
        err = budget(fn.__name__, kw)
        if err:
            return err
        try:
            res = fn(**kw)
        except Exception as e:  # tool errors go back to the agent, not crash the server
            res = {"error": str(e)}
        log(fn.__name__, kw, res)
        return json.dumps(res) + " " + remaining()
    return register(fn, wrapper)


def image_tool(fn):
    def wrapper(**kw):
        err = budget(fn.__name__, kw, image=True)
        if err:
            return err
        try:
            pic, note = fn(**kw)
        except Exception as e:
            state["images"] -= 1  # failed renders don't use up the image budget
            log(fn.__name__, kw, {"error": str(e)})
            return json.dumps({"error": str(e)}) + " " + remaining()
        log(fn.__name__, kw, note)
        return [pic, note + " " + remaining()]
    return register(fn, wrapper)


# ---------- tools ----------

@tool
def info():
    """Geometry and intensity summary of the scan. Call this first."""
    return {
        "shape_ijk": list(vol.shape),
        "spacing_mm": [round(float(s), 3) for s in sp],
        "coverage_mm": {AXIS_NAMES[a]: round(float(vol.shape[a] * sp[a]), 1) for a in range(3)},
        "acquired_plane": NORMAL_PLANE[ACQ_AXIS] + f" (slices stacked along {'ijk'[ACQ_AXIS]})",
        "axes": "i increases toward patient Right, j toward Anterior, k toward Superior. "
                "Images are shown radiologically (patient Right on image left) with true mm proportions.",
        "intensity_nonzero_percentiles": {str(p): round(float(np.percentile(NZ, p)), 1)
                                          for p in (1, 5, 25, 50, 75, 95, 99, 99.9)},
    }


@tool
def stats(box: list[int] | None = None, mask: str | None = None):
    """Intensity statistics inside a box [i0,i1,j0,j1,k0,k1] (end exclusive) and/or inside a named mask."""
    sel = np.zeros(vol.shape, bool)
    sel[box_slices(box)] = True
    if mask:
        sel &= get_mask(mask)
    v = vol[sel]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": round(float(v.mean()), 1), "std": round(float(v.std()), 1),
            "percentiles": {str(p): round(float(np.percentile(v, p)), 1) for p in (1, 5, 25, 50, 75, 95, 99)}}


@image_tool
def overview(plane: str = "axial", n: int = 16):
    """Montage of n slices (max 16) evenly spaced across the whole scan in one plane: axial, coronal or sagittal."""
    a = PLANE_NORMAL[plane]
    n = int(min(max(n, 1), 16))
    idx = sorted({int(round(x)) for x in np.linspace(0, vol.shape[a] - 1, n)})
    return render(a, idx, title=f"overview {plane}"), f"overview {plane}: slices {idx}"


@image_tool
def view(plane: str, center: list[int], fov_mm: float = 60.0, n_slices: int = 9,
         step_mm: float | None = None, mask: str | None = None):
    """Zoomed montage around center [i,j,k]: in-plane field of view fov_mm, n_slices (max 16) along the
    plane normal spaced step_mm apart (default: slice spacing, at least 2 mm). Optional mask outline."""
    a = PLANE_NORMAL[plane]
    n = int(min(max(n_slices, 1), 16))
    step_vox = max((step_mm if step_mm else max(sp[a], 2.0)) / sp[a], 1.0)
    offsets = (np.arange(n) - (n - 1) / 2) * step_vox
    idx = sorted({int(np.floor(center[a] + o + 0.5)) for o in offsets if 0 <= center[a] + o <= vol.shape[a] - 1})
    _, (h, v) = plane_slice(vol, a, 0)
    rh, rv = fov_mm / 2 / sp[h], fov_mm / 2 / sp[v]
    crop = (center[h] - rh, center[h] + rh, center[v] - rv, center[v] + rv)
    m = get_mask(mask) if mask else None
    return render(a, idx, crop, m, f"view {plane} around {center}"), f"view {plane}: slices {idx}"


@image_tool
def review(mask: str, plane: str | None = None, page: int = 1):
    """Outline montage covering the mask's full extent (+3 mm) along the plane normal, cropped to the mask
    (+15 mm). Up to 24 panels per page; use page=2,... if the result says there are more pages.
    Default plane: the acquired plane."""
    m = get_mask(mask)
    if not m.any():
        raise ValueError("mask is empty")
    a = PLANE_NORMAL[plane] if plane else ACQ_AXIS
    idx3 = np.argwhere(m)
    lo, hi = idx3.min(0), idx3.max(0)
    pad = np.ceil(3.0 / sp[a])
    all_idx = list(range(int(max(0, lo[a] - pad)), int(min(vol.shape[a] - 1, hi[a] + pad)) + 1))
    pages = [all_idx[q:q + 24] for q in range(0, len(all_idx), 24)]
    page = int(min(max(page, 1), len(pages)))
    _, (h, v) = plane_slice(vol, a, 0)
    mh, mv = 15.0 / sp[h], 15.0 / sp[v]
    crop = (lo[h] - mh, hi[h] + mh, lo[v] - mv, hi[v] + mv)
    pic = render(a, pages[page - 1], crop, m, f"review '{mask}' ({NORMAL_PLANE[a]}) page {page}/{len(pages)}")
    return pic, f"review {NORMAL_PLANE[a]}: page {page}/{len(pages)}, slices {pages[page - 1][0]}-{pages[page - 1][-1]}"


@tool
def threshold(out: str, lo: float, hi: float, box: list[int] | None = None):
    """New mask `out` = voxels with lo <= intensity <= hi, restricted to an optional box."""
    m = np.zeros(vol.shape, bool)
    s = box_slices(box)
    m[s] = (vol[s] >= lo) & (vol[s] <= hi)
    masks[out] = m
    return {"mask": out, **describe(m)}


@tool
def grow(out: str, seed: list[int], lo: float, hi: float, box: list[int] | None = None):
    """New mask `out` = region connected (6-neighbourhood) to seed [i,j,k] with lo <= intensity <= hi,
    restricted to an optional box."""
    allowed = np.zeros(vol.shape, bool)
    s = box_slices(box)
    allowed[s] = (vol[s] >= lo) & (vol[s] <= hi)
    seed = tuple(int(x) for x in seed)
    if not allowed[seed]:
        raise ValueError(f"seed intensity {float(vol[seed])} is outside [{lo}, {hi}] or outside the box")
    lab, _ = ndi.label(allowed)
    masks[out] = lab == lab[seed]
    return {"mask": out, "seed_intensity": float(vol[seed]), **describe(masks[out])}


@tool
def components(mask: str, out: str | None = None, keep: list[int] | None = None):
    """List connected components of `mask` (largest first, numbered from 1). With `out` and `keep`,
    store only the listed component numbers as a new mask."""
    lab, n = ndi.label(get_mask(mask), structure=ndi.generate_binary_structure(3, 3))
    order = np.argsort(np.bincount(lab.ravel())[1:])[::-1] + 1
    comps = []
    for rank, c in enumerate(order[:10], start=1):
        idx = np.argwhere(lab == c)
        comps.append({"component": rank, "voxels": len(idx), "volume_ml": round(len(idx) * VOX_ML, 3),
                      "bbox_ijk": [int(x) for x in np.r_[idx.min(0), idx.max(0)]],
                      "centroid_ijk": [round(float(x), 1) for x in idx.mean(0)]})
    res = {"n_components": int(n), "components": comps}
    if out and keep:
        masks[out] = np.isin(lab, [order[r - 1] for r in keep if 1 <= r <= len(order)])
        res["kept"] = {"mask": out, **describe(masks[out])}
    return res


@tool
def morph(mask: str, out: str, op: str, radius_mm: float = 1.0, within: str | None = None):
    """Morphology with a ball of radius_mm (respects voxel spacing): op = open | close | dilate | erode |
    fill (fill holes slice-wise in the acquired plane). Optional `within` restricts the result to another mask."""
    m = get_mask(mask)
    if op == "fill":
        a = ACQ_AXIS
        r = np.stack([ndi.binary_fill_holes(np.take(m, q, axis=a)) for q in range(m.shape[a])], axis=a)
    else:
        f = {"open": ndi.binary_opening, "close": ndi.binary_closing,
             "dilate": ndi.binary_dilation, "erode": ndi.binary_erosion}[op]
        r = f(m, structure=ball(radius_mm))
    if within:
        r &= get_mask(within)
    masks[out] = r
    return {"mask": out, **describe(r)}


@tool
def exclude_box(mask: str, out: str, box: list[int]):
    """New mask `out` = `mask` with all voxels inside box [i0,i1,j0,j1,k0,k1] (end exclusive) removed."""
    r = get_mask(mask).copy()
    r[box_slices(box)] = False
    masks[out] = r
    return {"mask": out, **describe(r)}


@tool
def union(a: str, b: str, out: str):
    """New mask `out` = union of masks a and b."""
    masks[out] = get_mask(a) | get_mask(b)
    return {"mask": out, **describe(masks[out])}


@mcp.tool()
def save(mask: str) -> str:
    """Save `mask` as the final segmentation (label.nii.gz on the original scan grid). Can be called again
    to overwrite; the last saved mask is the answer. Allowed even when the budget is exhausted."""
    state["calls"] += 1
    try:
        m = get_mask(mask)
        back = ornt.apply_orientation(m.astype(np.uint8), ornt.ornt_transform(ornt.io_orientation(img.affine), orig_ornt))
        out = nib.Nifti1Image(back, orig.affine, orig.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, OUT / "label.nii.gz")
        state["saved"] = True
        res = {"saved": mask, **describe(m)}
    except Exception as e:
        res = {"error": str(e)}
    log("save", {"mask": mask}, res)
    return json.dumps(res)


if __name__ == "__main__":
    mcp.run()
