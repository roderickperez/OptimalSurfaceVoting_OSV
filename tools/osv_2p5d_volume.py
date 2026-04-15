"""osv_2p5d_volume.py – per-section 2.5D OSV helper.

Two entry-points:
  run_volume_osv_2p5d()            – from pre-computed 3D seismic + FaultSeg volumes.
  run_volume_osv_2p5d_from_sgy()   – directly from SEG-Y; no pre-computed volume needed.
                                     Checkpoints each section for resumable runs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def normalize_section(section: np.ndarray) -> np.ndarray:
    section = np.nan_to_num(section.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p1, p99 = np.percentile(section, [1, 99])
    section = np.clip(section, p1, p99)
    scale = np.max(np.abs(section)) + 1.0e-6
    return section / scale


def sharpen_faultseg(section: np.ndarray, power: float) -> np.ndarray:
    section = np.nan_to_num(section.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    section = np.clip(section, 0.0, 1.0)
    if power != 1.0:
        section = np.power(section, power, dtype=np.float32)
    return np.clip(section, 0.0, 1.0)


def read_dat_2d(path: Path, n1: int, n2: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=">f4")
    if arr.size != n1 * n2:
        raise ValueError(f"Unexpected size for {path}: {arr.size}, expected {n1 * n2}")
    return arr.reshape(n2, n1).T.astype(np.float32)


def volume_section(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "inline":
        return volume[:, index, :]
    if axis == "crossline":
        return volume[:, :, index]
    raise ValueError(f"Unsupported axis {axis}")


def section_count(volume: np.ndarray, axis: str) -> int:
    if axis == "inline":
        return int(volume.shape[1])
    if axis == "crossline":
        return int(volume.shape[2])
    raise ValueError(f"Unsupported axis {axis}")


def insert_section(volume: np.ndarray, axis: str, index: int, section: np.ndarray) -> None:
    if axis == "inline":
        volume[:, index, :] = section
        return
    if axis == "crossline":
        volume[:, :, index] = section
        return
    raise ValueError(f"Unsupported axis {axis}")


def run_osv_on_section(
    project_root: Path,
    seismic_section: np.ndarray,
    fault_section: np.ndarray,
    tmp_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    n1, n2 = seismic_section.shape

    in_dat = tmp_dir / "input_slice.dat"
    fault_dat = tmp_dir / "fault_slice.dat"
    out_dir = tmp_dir / "osv_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    seismic_section.T.astype(">f4").tofile(in_dat)
    fault_section.T.astype(">f4").tofile(fault_dat)

    cmd = [
        "java",
        "--class-path",
        "libs/*:build/classes",
        "tools/OSV2DRunner.java",
        str(in_dat),
        str(out_dir),
        str(n1),
        str(n2),
        str(fault_dat),
    ]
    subprocess.run(
        cmd,
        cwd=str(project_root),
        text=True,
        capture_output=True,
        check=True,
    )

    fv = read_dat_2d(out_dir / "fv.dat", n1, n2)
    fvg = read_dat_2d(out_dir / "fvg.dat", n1, n2)
    fvt = read_dat_2d(out_dir / "fvt.dat", n1, n2)
    fvtg = read_dat_2d(out_dir / "fvtg.dat", n1, n2)
    return fv, fvg, fvt, fvtg


def run_volume_osv_2p5d(
    project_root: Path,
    seismic_volume_path: Path,
    faultseg_volume_path: Path,
    output_dir: Path,
    axis: str,
    sharpen_power: float,
    extract_threshold: float,
    max_sections: int | None = None,
) -> dict:
    seismic_volume = np.load(seismic_volume_path).astype(np.float32)
    faultseg_volume = np.load(faultseg_volume_path).astype(np.float32)
    if seismic_volume.shape != faultseg_volume.shape:
        raise ValueError(
            f"Seismic and FaultSeg volumes must match: {seismic_volume.shape} vs {faultseg_volume.shape}"
        )

    n1, n2, n3 = seismic_volume.shape
    total_sections = section_count(seismic_volume, axis)
    sections_to_run = total_sections if max_sections is None else min(total_sections, int(max_sections))

    fv_volume = np.zeros_like(faultseg_volume, dtype=np.float32)
    fvg_volume = np.zeros_like(faultseg_volume, dtype=np.float32)
    fvt_volume = np.zeros_like(faultseg_volume, dtype=np.float32)
    fvtg_volume = np.zeros_like(faultseg_volume, dtype=np.float32)
    fvtg_binary_volume = np.zeros_like(faultseg_volume, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root = output_dir / "tmp_sections"
    temp_root.mkdir(parents=True, exist_ok=True)

    for section_index in range(sections_to_run):
        seismic_section = normalize_section(volume_section(seismic_volume, axis, section_index))
        fault_section = sharpen_faultseg(volume_section(faultseg_volume, axis, section_index), sharpen_power)
        section_tmp = temp_root / f"{axis}_{section_index:04d}"

        fv, fvg, fvt, fvtg = run_osv_on_section(project_root, seismic_section, fault_section, section_tmp)
        fvtg_binary = (fvtg >= extract_threshold).astype(np.float32)

        insert_section(fv_volume, axis, section_index, fv)
        insert_section(fvg_volume, axis, section_index, fvg)
        insert_section(fvt_volume, axis, section_index, fvt)
        insert_section(fvtg_volume, axis, section_index, fvtg)
        insert_section(fvtg_binary_volume, axis, section_index, fvtg_binary)

        if section_index == 0 or (section_index + 1) % 25 == 0 or section_index + 1 == sections_to_run:
            print(f"Processed {section_index + 1}/{sections_to_run} {axis} sections")

    np.save(output_dir / "fv_volume.npy", fv_volume)
    np.save(output_dir / "fvg_volume.npy", fvg_volume)
    np.save(output_dir / "fvt_volume.npy", fvt_volume)
    np.save(output_dir / "fvtg_volume.npy", fvtg_volume)
    np.save(output_dir / "fvtg_binary_volume.npy", fvtg_binary_volume)

    metadata = {
        "axis": axis,
        "shape": [int(n1), int(n2), int(n3)],
        "sharpen_power": float(sharpen_power),
        "extract_threshold": float(extract_threshold),
        "sections_processed": int(sections_to_run),
        "total_sections": int(total_sections),
        "fv_volume_npy": str(output_dir / "fv_volume.npy"),
        "fvg_volume_npy": str(output_dir / "fvg_volume.npy"),
        "fvt_volume_npy": str(output_dir / "fvt_volume.npy"),
        "fvtg_volume_npy": str(output_dir / "fvtg_volume.npy"),
        "fvtg_binary_volume_npy": str(output_dir / "fvtg_binary_volume.npy"),
    }
    (output_dir / "osv_2p5d_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


# ─── Mode 2: directly from SEG-Y (no full-volume pre-required) ──────────────

def _add_faultseg_to_path(faultseg_dir: Path) -> None:
    s = str(Path(faultseg_dir).resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _centered_indices(total: int, center: int, window: int) -> np.ndarray:
    """Return up to `window` contiguous indices centered on `center` in [0, total)."""
    half = window // 2
    start = max(0, center - half)
    end = min(total, start + window)
    start = max(0, end - window)  # shift left if we hit the right boundary
    return np.arange(start, end, dtype=int)


def _load_segy_volume(sgy_path: Path) -> tuple:
    """Load entire SEG-Y into float32[n_samples, n_inlines, n_xlines].

    Uses chunked trace reading (5000 traces/chunk) for memory efficiency.
    Returns (volume, unique_inline_ids, unique_xline_ids).
    """
    import segyio

    with segyio.open(str(sgy_path), mode="r", ignore_geometry=True) as f:
        f.mmap()
        ilines_hdr = np.asarray(f.attributes(segyio.TraceField.INLINE_3D)[:])
        xlines_hdr = np.asarray(f.attributes(segyio.TraceField.CROSSLINE_3D)[:])
        n_samples = int(len(f.samples))
        n_traces  = int(f.tracecount)

        unique_il = np.unique(ilines_hdr)
        unique_xl = np.unique(xlines_hdr)
        il_pos = np.searchsorted(unique_il, ilines_hdr)
        xl_pos = np.searchsorted(unique_xl, xlines_hdr)
        n_il = int(len(unique_il))
        n_xl = int(len(unique_xl))

        print(f"  SEG-Y: {n_traces} traces · {n_samples} samples · "
              f"{n_il} inlines · {n_xl} xlines  "
              f"({n_samples * n_il * n_xl * 4 / 1e9:.2f} GB)")

        vol = np.zeros((n_samples, n_il, n_xl), dtype=np.float32)
        chunk = 5_000
        for t0 in range(0, n_traces, chunk):
            t1 = min(n_traces, t0 + chunk)
            raw = np.array([f.trace.raw[t] for t in range(t0, t1)], dtype=np.float32)
            vol[:, il_pos[t0:t1], xl_pos[t0:t1]] = raw.T
            if (t0 // chunk) % 20 == 0:
                print(f"    loaded {t1}/{n_traces} traces…", end="\r", flush=True)
        print(f"    loaded {n_traces}/{n_traces} traces   ")

    return vol, unique_il.astype(int), unique_xl.astype(int)


def _write_assembled(
    output_dir: Path,
    seismic_volume: np.ndarray,
    fvtg_vol: np.ndarray,
    fvtg_b_vol: np.ndarray,
    fvg_vol: np.ndarray,
    fault_vol: np.ndarray,
    axis: str,
    unique_il,
    unique_xl,
    sharpen_power: float,
    extract_threshold: float,
    sections_processed: int,
    total_sections: int,
) -> None:
    """Write assembled volumes + meta JSON to disk (used for both incremental and final saves)."""
    n_samples, n_inlines, n_xlines = seismic_volume.shape
    np.save(output_dir / "fvtg_volume.npy",        fvtg_vol)
    np.save(output_dir / "fvtg_binary_volume.npy", fvtg_b_vol)
    np.save(output_dir / "fvg_volume.npy",         fvg_vol)
    np.save(output_dir / "seismic_volume.npy",     seismic_volume)
    np.save(output_dir / "faultseg_volume.npy",    fault_vol)
    metadata = {
        "axis":                    axis,
        "shape":                   [int(n_samples), int(n_inlines), int(n_xlines)],
        "inline_ids":              unique_il.tolist(),
        "xline_ids":               unique_xl.tolist(),
        "sharpen_power":           float(sharpen_power),
        "extract_threshold":       float(extract_threshold),
        "sections_processed":      int(sections_processed),
        "total_sections":          int(total_sections),
        "fvtg_volume_npy":         str(output_dir / "fvtg_volume.npy"),
        "fvtg_binary_volume_npy":  str(output_dir / "fvtg_binary_volume.npy"),
        "fvg_volume_npy":          str(output_dir / "fvg_volume.npy"),
        "seismic_volume_npy":      str(output_dir / "seismic_volume.npy"),
        "faultseg_volume_npy":     str(output_dir / "faultseg_volume.npy"),
    }
    meta_path = output_dir / "osv_2p5d_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_volume_osv_2p5d_from_sgy(
    project_root: Path,
    sgy_path: Path,
    faultseg_dir: Path,
    output_dir: Path,
    axis: str,
    sharpen_power: float,
    extract_threshold: float,
    max_sections=None,
    section_window: int = 32,
    model_path=None,
    device_name=None,
) -> dict:
    """2.5D OSV computed directly from a SEG-Y file.

    No pre-computed FaultSeg volume is required.  Pipeline:
      1. Load full seismic into RAM from SEG-Y once.
      2. Load FaultSeg model once.
      3. Per section: extract sub-volume window → FaultSeg → extract section
         fault slice → OSV2DRunner → checkpoint.
      4. Assemble into 3D volumes and write osv_2p5d_meta.json.
    Resumable: sections that already have a checkpoint file are skipped.
    """
    _add_faultseg_to_path(Path(faultseg_dir))

    import torch
    from predict_sgy_inline import latest_model_path, load_model, run_tiled_prediction

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sections_dir = output_dir / "sections"
    sections_dir.mkdir(exist_ok=True)

    # 1. Seismic volume
    print("Loading seismic volume from SEG-Y…")
    seismic_volume, unique_il, unique_xl = _load_segy_volume(Path(sgy_path))
    n_samples, n_inlines, n_xlines = seismic_volume.shape

    if axis == "inline":
        section_ids = unique_il
    elif axis == "crossline":
        section_ids = unique_xl
    else:
        raise ValueError(f"axis must be 'inline' or 'crossline', got {axis!r}")

    n_total  = len(section_ids)
    n_to_run = n_total if max_sections is None else min(n_total, int(max_sections))

    # 2. FaultSeg model
    faultseg_dir = Path(faultseg_dir)
    resolved_model = Path(model_path) if model_path else latest_model_path(faultseg_dir / "model")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(resolved_model, device)
    model.eval()
    print(f"FaultSeg model : {resolved_model.name}")
    print(f"Device         : {device}")
    print(f"Axis           : {axis}  |  sections to process: {n_to_run}/{n_total}")
    if max_sections is not None:
        print(f"  (partial run — set max_sections=None for full volume)")

    # 3. Output arrays
    fvtg_vol   = np.zeros((n_samples, n_inlines, n_xlines), dtype=np.float32)
    fvtg_b_vol = np.zeros((n_samples, n_inlines, n_xlines), dtype=np.float32)
    fvg_vol    = np.zeros((n_samples, n_inlines, n_xlines), dtype=np.float32)
    fault_vol  = np.zeros((n_samples, n_inlines, n_xlines), dtype=np.float32)

    # 4. Section loop
    for i in range(n_to_run):
        section_id = int(section_ids[i])
        fvtg_cache = sections_dir / f"{axis}_{i:04d}_fvtg.npy"

        if fvtg_cache.exists():
            fvtg_s   = np.load(fvtg_cache)
            fvtg_b_s = np.load(sections_dir / f"{axis}_{i:04d}_fvtg_binary.npy")
            fvg_s    = np.load(sections_dir / f"{axis}_{i:04d}_fvg.npy")
            fa_cache = sections_dir / f"{axis}_{i:04d}_faultseg.npy"
            fa_s     = np.load(fa_cache) if fa_cache.exists() else np.zeros_like(fvtg_s)
        else:
            # 4a. Extract seismic sub-volume
            if axis == "inline":
                win_idx = _centered_indices(n_inlines, i, section_window)
                sub_vol = seismic_volume[:, win_idx, :].copy()
                tgt_pos = int(np.searchsorted(win_idx, i))
            else:
                win_idx = _centered_indices(n_xlines, i, section_window)
                sub_vol = seismic_volume[:, :, win_idx].copy()
                tgt_pos = int(np.searchsorted(win_idx, i))

            # 4b. FaultSeg
            with torch.no_grad():
                fault_sub = run_tiled_prediction(sub_vol, model, device)

            # 4c. Extract section
            if axis == "inline":
                seis_s  = normalize_section(sub_vol[:, tgt_pos, :].copy())
                fau_raw = np.clip(fault_sub[:, tgt_pos, :], 0.0, 1.0).astype(np.float32)
            else:
                seis_s  = normalize_section(sub_vol[:, :, tgt_pos].copy())
                fau_raw = np.clip(fault_sub[:, :, tgt_pos], 0.0, 1.0).astype(np.float32)
            fa_s = sharpen_faultseg(fau_raw, sharpen_power)

            # 4d. OSV
            tmp_dir = output_dir / "tmp_sections" / f"{axis}_{i:04d}"
            _, fvg_s, _, fvtg_s = run_osv_on_section(project_root, seis_s, fa_s, tmp_dir)
            fvtg_b_s = (fvtg_s >= extract_threshold).astype(np.float32)

            # 4e. Checkpoint
            np.save(fvtg_cache,                                             fvtg_s)
            np.save(sections_dir / f"{axis}_{i:04d}_fvtg_binary.npy",      fvtg_b_s)
            np.save(sections_dir / f"{axis}_{i:04d}_fvg.npy",              fvg_s)
            np.save(sections_dir / f"{axis}_{i:04d}_faultseg.npy",         fa_s)

        # 4f. Insert into assembled volumes
        if axis == "inline":
            fvtg_vol[:,   i, :] = fvtg_s
            fvtg_b_vol[:, i, :] = fvtg_b_s
            fvg_vol[:,    i, :] = fvg_s
            fault_vol[:,  i, :] = fa_s
        else:
            fvtg_vol[:,   :, i] = fvtg_s
            fvtg_b_vol[:, :, i] = fvtg_b_s
            fvg_vol[:,    :, i] = fvg_s
            fault_vol[:,  :, i] = fa_s

        if i == 0 or (i + 1) % 10 == 0 or i + 1 == n_to_run:
            print(f"  [{i + 1}/{n_to_run}] {axis} id={section_id}", flush=True)

        # 4g. Incremental checkpoint: write assembled volumes + meta every 50 sections
        if (i + 1) % 50 == 0 and i + 1 < n_to_run:
            _write_assembled(
                output_dir, seismic_volume, fvtg_vol, fvtg_b_vol, fvg_vol, fault_vol,
                axis, unique_il, unique_xl, sharpen_power, extract_threshold,
                sections_processed=i + 1, total_sections=n_total,
            )
            print(f"  → incremental save at {i + 1} sections", flush=True)

    # 5. Save assembled volumes
    print("Saving assembled volumes…")
    _write_assembled(
        output_dir, seismic_volume, fvtg_vol, fvtg_b_vol, fvg_vol, fault_vol,
        axis, unique_il, unique_xl, sharpen_power, extract_threshold,
        sections_processed=n_to_run, total_sections=n_total,
    )
    meta_path = output_dir / "osv_2p5d_meta.json"
    print(f"Done.  Metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2.5D OSV (from pre-computed volumes or directly from SEG-Y)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # volumes sub-command (legacy)
    p_vol = sub.add_parser("volumes", help="Run from pre-computed seismic/FaultSeg volumes.")
    p_vol.add_argument("--project-root",      type=Path, required=True)
    p_vol.add_argument("--seismic-volume",    type=Path, required=True)
    p_vol.add_argument("--faultseg-volume",   type=Path, required=True)
    p_vol.add_argument("--output-dir",        type=Path, required=True)
    p_vol.add_argument("--axis",              choices=["inline", "crossline"], default="inline")
    p_vol.add_argument("--sharpen-power",     type=float, default=2.0)
    p_vol.add_argument("--extract-threshold", type=float, default=0.20)
    p_vol.add_argument("--max-sections",      type=int,   default=None)

    # segy sub-command
    p_sgy = sub.add_parser("segy", help="Run directly from a SEG-Y file.")
    p_sgy.add_argument("--project-root",      type=Path, required=True)
    p_sgy.add_argument("--sgy-path",          type=Path, required=True)
    p_sgy.add_argument("--faultseg-dir",      type=Path, required=True)
    p_sgy.add_argument("--output-dir",        type=Path, required=True)
    p_sgy.add_argument("--axis",              choices=["inline", "crossline"], default="inline")
    p_sgy.add_argument("--sharpen-power",     type=float, default=2.0)
    p_sgy.add_argument("--extract-threshold", type=float, default=0.20)
    p_sgy.add_argument("--max-sections",      type=int,   default=None)
    p_sgy.add_argument("--section-window",    type=int,   default=32)
    p_sgy.add_argument("--model-path",        type=Path,  default=None)
    p_sgy.add_argument("--device",            type=str,   default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "volumes":
        meta = run_volume_osv_2p5d(
            project_root=args.project_root,
            seismic_volume_path=args.seismic_volume,
            faultseg_volume_path=args.faultseg_volume,
            output_dir=args.output_dir,
            axis=args.axis,
            sharpen_power=args.sharpen_power,
            extract_threshold=args.extract_threshold,
            max_sections=args.max_sections,
        )
    else:  # segy
        meta = run_volume_osv_2p5d_from_sgy(
            project_root=args.project_root,
            sgy_path=args.sgy_path,
            faultseg_dir=args.faultseg_dir,
            output_dir=args.output_dir,
            axis=args.axis,
            sharpen_power=args.sharpen_power,
            extract_threshold=args.extract_threshold,
            max_sections=args.max_sections,
            section_window=args.section_window,
            model_path=args.model_path,
            device_name=args.device,
        )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()