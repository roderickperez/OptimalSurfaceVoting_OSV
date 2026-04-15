import argparse
import json
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch

from unet3_pytorch import unet


PATCH_N1 = 128
PATCH_N2 = 128
PATCH_N3 = 128
OVERLAP = 12


def segy_geometry(sgy_path: Path):
    with segyio.open(str(sgy_path), mode="r", ignore_geometry=True) as segy_file:
        segy_file.mmap()
        ilines = np.asarray(segy_file.attributes(segyio.TraceField.INLINE_3D)[:])
        xlines = np.asarray(segy_file.attributes(segyio.TraceField.CROSSLINE_3D)[:])
        sample_count = len(segy_file.samples)
    return {
        "ilines": ilines,
        "xlines": xlines,
        "unique_ilines": np.unique(ilines).astype(int),
        "unique_xlines": np.unique(xlines).astype(int),
        "sample_count": int(sample_count),
    }


def latest_model_path(model_dir: Path) -> Path:
    models = sorted(model_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime)
    if not models:
        raise FileNotFoundError(f"No .pth models found in {model_dir}")
    return models[-1]


def load_model(model_path: Path, device: torch.device):
    model = unet(input_size=(PATCH_N1, PATCH_N2, PATCH_N3), pretrained_weights=None)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def choose_inline(ilines: np.ndarray) -> int:
    unique_il, counts = np.unique(ilines, return_counts=True)
    if unique_il.size == 0:
        raise RuntimeError("No inline headers found in SEG-Y file.")
    return int(unique_il[int(np.argmax(counts))])


def centered_window(values: np.ndarray, center_value: int, max_count: int) -> np.ndarray:
    values = np.asarray(np.sort(values))
    max_count = min(max_count, values.size)
    center_index = int(np.where(values == center_value)[0][0])
    start = max(0, center_index - max_count // 2)
    start = min(start, values.size - max_count)
    end = start + max_count
    return values[start:end]


def create_blend_mask(overlap: int, shape1: int, shape2: int, shape3: int) -> np.ndarray:
    def ramp_window(size: int, overlap_size: int) -> np.ndarray:
        if overlap_size <= 0:
            return np.ones(size, dtype=np.float32)
        overlap_size = min(overlap_size, size // 2)
        window = np.ones(size, dtype=np.float32)
        if overlap_size > 0:
            # Keep non-zero edge weights so boundary inlines/slices are not nulled.
            edge_floor = 0.1
            window[:overlap_size] = np.linspace(edge_floor, 1.0, overlap_size, dtype=np.float32)
            window[-overlap_size:] = np.linspace(1.0, edge_floor, overlap_size, dtype=np.float32)
        return window

    w1 = ramp_window(shape1, overlap)
    w2 = ramp_window(shape2, overlap)
    w3 = ramp_window(shape3, overlap)
    return np.einsum("i,j,k->ijk", w1, w2, w3, dtype=np.float32)


def padded_axis(size: int, patch: int, stride: int) -> int:
    if size <= patch:
        return patch
    extra = (stride - (size - patch) % stride) % stride
    return size + extra


def build_inline_subvolume(sgy_path: Path, target_inline: int | None = None, inline_window: int = 128):
    geometry = segy_geometry(sgy_path)
    ilines = geometry["ilines"]
    xlines = geometry["xlines"]
    sample_count = geometry["sample_count"]

    with segyio.open(str(sgy_path), mode="r", ignore_geometry=True) as segy_file:
        segy_file.mmap()

        if target_inline is None:
            target_inline = choose_inline(ilines)

        unique_il = np.unique(ilines)
        selected_il = centered_window(unique_il, target_inline, inline_window)

        target_indices = np.where(ilines == target_inline)[0]
        if target_indices.size == 0:
            raise RuntimeError(f"Inline {target_inline} not found in {sgy_path}")

        target_xlines = np.sort(np.unique(xlines[target_indices]))
        xline_to_column = {int(xline): index for index, xline in enumerate(target_xlines)}
        target_inline_position = int(np.where(selected_il == target_inline)[0][0])

        volume = np.zeros((sample_count, selected_il.size, target_xlines.size), dtype=np.float32)
        trace_counts = np.zeros((selected_il.size, target_xlines.size), dtype=np.int32)

        for inline_position, inline_id in enumerate(selected_il):
            indices = np.where(ilines == inline_id)[0]
            ordered = indices[np.argsort(xlines[indices])]
            for trace_index in ordered:
                xline_id = int(xlines[trace_index])
                column = xline_to_column.get(xline_id)
                if column is None:
                    continue
                volume[:, inline_position, column] = np.asarray(segy_file.trace[int(trace_index)], dtype=np.float32)
                trace_counts[inline_position, column] += 1

    return {
        "volume": volume,
        "target_inline": int(target_inline),
        "target_inline_position": target_inline_position,
        "selected_inlines": selected_il.astype(int),
        "target_xlines": target_xlines.astype(int),
        "trace_counts": trace_counts,
    }


def build_volume_for_inlines(
    sgy_path: Path,
    selected_inlines: np.ndarray,
    target_xlines: np.ndarray | None = None,
):
    geometry = segy_geometry(sgy_path)
    ilines = geometry["ilines"]
    xlines = geometry["xlines"]
    sample_count = geometry["sample_count"]
    if target_xlines is None:
        target_xlines = geometry["unique_xlines"]

    xline_to_column = {int(xline): index for index, xline in enumerate(target_xlines)}
    volume = np.zeros((sample_count, selected_inlines.size, target_xlines.size), dtype=np.float32)

    with segyio.open(str(sgy_path), mode="r", ignore_geometry=True) as segy_file:
        segy_file.mmap()
        for inline_position, inline_id in enumerate(selected_inlines):
            indices = np.where(ilines == inline_id)[0]
            ordered = indices[np.argsort(xlines[indices])]
            for trace_index in ordered:
                xline_id = int(xlines[trace_index])
                column = xline_to_column.get(xline_id)
                if column is None:
                    continue
                volume[:, inline_position, column] = np.asarray(segy_file.trace[int(trace_index)], dtype=np.float32)

    return volume, target_xlines.astype(int)


def inline_chunk_weights(length: int, overlap: int) -> np.ndarray:
    if overlap <= 0 or length <= 1:
        return np.ones(length, dtype=np.float32)
    overlap = min(overlap, length // 2)
    weights = np.ones(length, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.25, 1.0, overlap, dtype=np.float32)
        weights[:overlap] = np.maximum(weights[:overlap], ramp)
        weights[-overlap:] = np.maximum(weights[-overlap:], ramp[::-1])
    return weights


def run_tiled_prediction(volume: np.ndarray, model, device: torch.device) -> np.ndarray:
    m1, m2, m3 = volume.shape
    stride = (PATCH_N1 - OVERLAP, PATCH_N2 - OVERLAP, PATCH_N3 - OVERLAP)

    padded_shape = (
        padded_axis(m1, PATCH_N1, stride[0]),
        padded_axis(m2, PATCH_N2, stride[1]),
        padded_axis(m3, PATCH_N3, stride[2]),
    )
    padded = np.pad(
        volume,
        ((0, padded_shape[0] - m1), (0, padded_shape[1] - m2), (0, padded_shape[2] - m3)),
        mode="constant",
        constant_values=0,
    )

    prediction = np.zeros(padded_shape, dtype=np.float32)
    weights = np.zeros(padded_shape, dtype=np.float32)
    blend = create_blend_mask(OVERLAP, PATCH_N1, PATCH_N2, PATCH_N3)

    k1_range = range(0, padded_shape[0] - PATCH_N1 + 1, stride[0])
    k2_range = range(0, padded_shape[1] - PATCH_N2 + 1, stride[1])
    k3_range = range(0, padded_shape[2] - PATCH_N3 + 1, stride[2])

    for k1 in k1_range:
        for k2 in k2_range:
            for k3 in k3_range:
                patch = padded[k1:k1 + PATCH_N1, k2:k2 + PATCH_N2, k3:k3 + PATCH_N3]
                patch = (patch - patch.mean()) / (patch.std() + 1.0e-8)
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                with torch.no_grad():
                    patch_prediction = model(tensor).squeeze().detach().cpu().numpy().astype(np.float32)
                prediction[k1:k1 + PATCH_N1, k2:k2 + PATCH_N2, k3:k3 + PATCH_N3] += patch_prediction * blend
                weights[k1:k1 + PATCH_N1, k2:k2 + PATCH_N2, k3:k3 + PATCH_N3] += blend

    averaged = np.divide(prediction, weights, out=np.zeros_like(prediction), where=weights > 0)
    return averaged[:m1, :m2, :m3]


def save_outputs(output_dir: Path, seismic_section: np.ndarray, fault_section: np.ndarray, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    inline_npy = output_dir / "faultseg_inline.npy"
    inline_dat = output_dir / "faultseg_inline.dat"
    meta_json = output_dir / "faultseg_inline_meta.json"
    preview_png = output_dir / "faultseg_inline_preview.png"

    np.save(inline_npy, fault_section.astype(np.float32))
    fault_section.T.astype(">f4").tofile(inline_dat)
    meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(seismic_section, cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].set_title("Seismic Inline")
    axes[0].set_xlabel("Trace index")
    axes[0].set_ylabel("Sample index")

    axes[1].imshow(fault_section, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("FaultSeg Prediction")
    axes[1].set_xlabel("Trace index")
    axes[1].set_ylabel("Sample index")

    fig.savefig(preview_png, dpi=170)
    plt.close(fig)

    return {
        "inline_npy": str(inline_npy),
        "inline_dat": str(inline_dat),
        "meta_json": str(meta_json),
        "preview_png": str(preview_png),
    }


def save_volume_outputs(
    output_dir: Path,
    fault_volume_path: Path,
    seismic_volume_path: Path,
    geometry: dict,
    metadata: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_json = output_dir / "faultseg_volume_meta.json"
    preview_png = output_dir / "faultseg_volume_preview.png"
    metadata.update(
        {
            "fault_volume_npy": str(fault_volume_path),
            "seismic_volume_npy": str(seismic_volume_path),
            "meta_json": str(meta_json),
            "preview_png": str(preview_png),
        }
    )
    meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fault_volume = np.load(fault_volume_path, mmap_mode="r")
    seismic_volume = np.load(seismic_volume_path, mmap_mode="r")
    mid_i2 = fault_volume.shape[1] // 2
    mid_i3 = fault_volume.shape[2] // 2
    mid_i1 = fault_volume.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(seismic_volume[:, mid_i2, :], cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].imshow(np.ma.masked_where(fault_volume[:, mid_i2, :] <= 0.05, fault_volume[:, mid_i2, :]), cmap="inferno", aspect="auto", interpolation="nearest", alpha=0.8)
    axes[0].set_title(f"Inline {geometry['unique_ilines'][mid_i2]}")

    axes[1].imshow(seismic_volume[:, :, mid_i3], cmap="gray", aspect="auto", interpolation="nearest")
    axes[1].imshow(np.ma.masked_where(fault_volume[:, :, mid_i3] <= 0.05, fault_volume[:, :, mid_i3]), cmap="inferno", aspect="auto", interpolation="nearest", alpha=0.8)
    axes[1].set_title(f"Crossline {geometry['unique_xlines'][mid_i3]}")

    axes[2].imshow(seismic_volume[mid_i1, :, :], cmap="gray", aspect="auto", interpolation="nearest")
    axes[2].imshow(np.ma.masked_where(fault_volume[mid_i1, :, :] <= 0.05, fault_volume[mid_i1, :, :]), cmap="inferno", aspect="auto", interpolation="nearest", alpha=0.8)
    axes[2].set_title(f"Timeslice {mid_i1}")

    for ax in axes:
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")

    fig.savefig(preview_png, dpi=170)
    plt.close(fig)
    return metadata


def run_prediction(
    sgy_path: Path,
    output_dir: Path,
    model_path: Path | None = None,
    target_inline: int | None = None,
    inline_window: int = 128,
    device_name: str | None = None,
):
    repo_dir = Path(__file__).resolve().parent
    model_path = model_path or latest_model_path(repo_dir / "model")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    batch = build_inline_subvolume(sgy_path, target_inline=target_inline, inline_window=inline_window)
    model = load_model(model_path, device)
    fault_volume = run_tiled_prediction(batch["volume"], model, device)

    inline_index = batch["target_inline_position"]
    seismic_section = batch["volume"][:, inline_index, :]
    fault_section = np.clip(fault_volume[:, inline_index, :], 0.0, 1.0).astype(np.float32)

    metadata = {
        "sgy_path": str(sgy_path),
        "model_path": str(model_path),
        "device": str(device),
        "target_inline": int(batch["target_inline"]),
        "inline_index_in_window": int(inline_index),
        "inline_window_size": int(batch["selected_inlines"].size),
        "n1": int(fault_section.shape[0]),
        "n2": int(fault_section.shape[1]),
    }
    metadata.update(save_outputs(output_dir, seismic_section, fault_section, metadata))
    return metadata


def run_full_volume_prediction(
    sgy_path: Path,
    output_dir: Path,
    model_path: Path | None = None,
    chunk_inlines: int = 128,
    chunk_overlap: int = 32,
    device_name: str | None = None,
):
    repo_dir = Path(__file__).resolve().parent
    model_path = model_path or latest_model_path(repo_dir / "model")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    geometry = segy_geometry(sgy_path)
    unique_ilines = geometry["unique_ilines"]
    unique_xlines = geometry["unique_xlines"]
    n1 = geometry["sample_count"]
    n2 = unique_ilines.size
    n3 = unique_xlines.size

    output_dir.mkdir(parents=True, exist_ok=True)
    seismic_volume_path = output_dir / "seismic_volume.npy"
    fault_volume_path = output_dir / "faultseg_volume.npy"
    weight_volume_path = output_dir / "faultseg_volume_weights.npy"

    if seismic_volume_path.exists() and fault_volume_path.exists() and (output_dir / "faultseg_volume_meta.json").exists():
        meta = json.loads((output_dir / "faultseg_volume_meta.json").read_text(encoding="utf-8"))
        return meta

    model = load_model(model_path, device)

    seismic_memmap = np.lib.format.open_memmap(seismic_volume_path, mode="w+", dtype=np.float32, shape=(n1, n2, n3))
    fault_acc = np.lib.format.open_memmap(fault_volume_path, mode="w+", dtype=np.float32, shape=(n1, n2, n3))
    weight_acc = np.lib.format.open_memmap(weight_volume_path, mode="w+", dtype=np.float32, shape=(n1, n2, n3))
    fault_acc[:] = 0.0
    weight_acc[:] = 0.0

    step = max(1, chunk_inlines - chunk_overlap)
    for start in range(0, n2, step):
        end = min(n2, start + chunk_inlines)
        selected_inlines = unique_ilines[start:end]
        seismic_chunk, _ = build_volume_for_inlines(sgy_path, selected_inlines, unique_xlines)
        fault_chunk = np.clip(run_tiled_prediction(seismic_chunk, model, device), 0.0, 1.0)
        chunk_weights = inline_chunk_weights(end - start, chunk_overlap)[None, :, None]

        seismic_memmap[:, start:end, :] = seismic_chunk
        fault_acc[:, start:end, :] += fault_chunk * chunk_weights
        weight_acc[:, start:end, :] += chunk_weights

    np.divide(fault_acc, np.maximum(weight_acc, 1.0e-6), out=fault_acc)
    fault_acc.flush()
    seismic_memmap.flush()
    weight_acc.flush()
    os.remove(weight_volume_path)

    metadata = {
        "sgy_path": str(sgy_path),
        "model_path": str(model_path),
        "device": str(device),
        "n1": int(n1),
        "n2": int(n2),
        "n3": int(n3),
        "chunk_inlines": int(chunk_inlines),
        "chunk_overlap": int(chunk_overlap),
        "inline_ids": unique_ilines.tolist(),
        "xline_ids": unique_xlines.tolist(),
        "inline_min": int(unique_ilines.min()),
        "inline_max": int(unique_ilines.max()),
        "xline_min": int(unique_xlines.min()),
        "xline_max": int(unique_xlines.max()),
    }
    return save_volume_outputs(output_dir, fault_volume_path, seismic_volume_path, geometry, metadata)


def main():
    parser = argparse.ArgumentParser(description="Run PyTorch fault segmentation on a SEG-Y inline neighborhood.")
    parser.add_argument("--sgy", required=True, help="Path to SEG-Y file")
    parser.add_argument("--output-dir", required=True, help="Directory for prediction outputs")
    parser.add_argument("--model", default=None, help="Optional .pth model path")
    parser.add_argument("--inline", dest="target_inline", type=int, default=None, help="Inline id to extract")
    parser.add_argument("--inline-window", type=int, default=128, help="Number of neighboring inlines for 3D prediction")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu")
    parser.add_argument("--full-volume", action="store_true", help="Predict the full 3D volume and save faultseg_volume.npy")
    parser.add_argument("--chunk-inlines", type=int, default=128, help="Inline chunk size for full-volume prediction")
    parser.add_argument("--chunk-overlap", type=int, default=32, help="Inline overlap between chunks for full-volume prediction")
    args = parser.parse_args()

    if args.full_volume:
        result = run_full_volume_prediction(
            sgy_path=Path(args.sgy),
            output_dir=Path(args.output_dir),
            model_path=Path(args.model) if args.model else None,
            chunk_inlines=args.chunk_inlines,
            chunk_overlap=args.chunk_overlap,
            device_name=args.device,
        )
    else:
        result = run_prediction(
            sgy_path=Path(args.sgy),
            output_dir=Path(args.output_dir),
            model_path=Path(args.model) if args.model else None,
            target_inline=args.target_inline,
            inline_window=args.inline_window,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()