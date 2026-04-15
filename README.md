# Optimal Surface Voting (OSV) for Automatic Fault Interpretation

This project is a fork of the original work by **Xinming Wu** (University of Texas at Austin). It provides tools for both 2D and 3D fault interpretation using optimal surface/path voting algorithms.

## Original Author & Reference
- **Original Author**: [Xinming Wu](http://www.jsg.utexas.edu/wu/)
- **Original Repository**: [xinwucwp/osv](https://github.com/xinwucwp/osv)
- **Publication**: [Automatic fault interpretation with optimal surface voting](http://www.jsg.utexas.edu/wu/files/wu2018automaticFaultInterpretationWithOptimalSurfaceVotingLow.pdf) (Geophysics, 2018)

If you use this work, please cite:
```bibtex
@article{wu2018automatic,
    author = {Xinming Wu and Sergey Fomel},
    title = {Automatic fault interpretation with optimal surface voting},
    journal = {GEOPHYSICS},
    volume = {83},
    issue = {5},
    pages = {O67-O82},
    year = {2018},
    doi = {10.1190/GEO2018-0115.1},
    URL = {https://library.seg.org/doi/abs/10.1190/geo2018-0115.1},
}
```

## Description and Use

### Core Algorithms
1.  **OptimalSurfaceVoter**: Computes optimal voting surfaces, voting scores, and a final 3D voting score map. This is the heart of the 3D interpretation.
2.  **OptimalPathVoter**: The 2D counterpart, focused on computing optimal voting paths and 2D score maps.
3.  **FaultOrientScanner (2D/3D)**: Quickly scans for approximate fault orientations (strikes and dips) to guide the voting process.
4.  **FaultSkinner**: Automatically constructs fault skins (surfaces) from the final voting score map produced by the voter.
5.  **LocalOrientFilter**: Computes structure tensors and seismic planarity/linearity, which are used as input attributes (fault likelihood).

### How it works
The process typically involves:
1.  **Preprocessing**: Computing planarity from seismic data using a `LocalOrientFilter`.
2.  **Scanning**: Using `FaultOrientScanner` to find initial fault orientations.
3.  **Voting**: Running `OptimalSurfaceVoter` (3D) or `OptimalPathVoter` (2D) to enhance fault features and suppress noise.
4.  **Thinning & Skinning**: Extracting clear fault surfaces from the voting scores.

## Languages and Architecture
- **Core Algorithms**: Written in **Java** (located in `src/osv/*.java`).
- **Demos & Utilities**: Written in **Jython** (Python syntax running on Java, located in `src/osv/*.py`).
- **Modern Port (In Progress)**: Core algorithms are being ported to **Standard Python** using **PyTorch** for GPU acceleration.

## Running the Original Code (Java/Jython)
To run the original implementation, you need a Java Runtime Environment (JRE) installed.

1.  Navigate to the source directory:
    ```bash
    cd src/osv
    ```
2.  Run a demo script using the provided `jy` helper:
    ```bash
    ./jy demoF3d.py
    ```

## Data Requirements
The original code processes seismic data in a specific binary format.

- **Format**: Raw binary (headerless).
- **Byte Order**: Big Endian.
- **File Extension**: Usually `.dat`.
- **Storage Location**: Store your data in the `./data/` directory, following the project's structure (e.g., `./data/3d/f3d/`).

### Example: F3 Dataset
To run the 3D demo (`demoF3d.py`), you need:
-   **File**: `xs.dat`
-   **Path**: `./data/3d/f3d/xs.dat`
-   **Dimensions**: n1=100, n2=400, n3=420

## Getting Started (Python Port)
The `uv` environment created in this project is for the **Standard Python (PyTorch)** version of the code.

### 1. Requirements
Ensure you have `uv` installed. If not, you can install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and Activate Environment
To isolate dependencies, create a virtual environment and activate it:
```bash
# Create the environment (already done if you followed my steps)
uv venv .venv --python 3.10

# Activate the environment
source .venv/bin/activate
```

### 3. Install Dependencies
If you need to reinstall or update libraries:
```bash
uv pip install -r requirements-osv.txt
```

---

## Python Port Implementation (In Progress)
The current effort involves porting these algorithms to a modern Python environment with GPU acceleration using PyTorch, specifically optimized for NVIDIA A5000 GPUs.

## Implementation Audit: FaultSeg + OSV

### Papers reviewed
- **Wu et al., 2019 (FaultSeg3D)**: uses a simplified 3D U-Net, trained with **200** synthetic seismic/fault pairs and validated on **20** pairs, on **128 x 128 x 128** cubes, with a **balanced binary cross-entropy** loss to handle severe class imbalance.
- **Wu and Fomel, 2018 (Optimal Surface Voting)**: starts from a fault attribute image, estimates orientations, computes optimal paths/surfaces through high-score control points, smooths/votes along those paths/surfaces, and then extracts thin fault features from the voting image.

### What the current notebooks actually implement
- `OSV_F3_SGY_Demo_2D.ipynb`: a **single-inline 2D workflow**.
- `OSV_F3_SGY_Demo.ipynb`: despite the name, this is also a **single-inline 2D workflow** with the same core pipeline as the 2D notebook.
- `OSV_F3_SGY_Demo_2p5D.ipynb`: a **2.5D workflow** that runs 2D OSV section-by-section and assembles the results into volume-shaped arrays.

### Is the FaultSeg output actually used?
Yes.

For the 2D and root demo notebooks, the pipeline is:
1. Read a seismic inline from SEG-Y.
2. Run `faultSeg_2019_pyTorch/prediction_pytorch_F3.ipynb` to generate `faultseg_inline.npy`.
3. Clip the fault probabilities to `[0, 1]`.
4. Apply power-law sharpening with `SHARPEN_POWER`.
5. Write the sharpened fault probability to `faultseg_inline_osv.dat`.
6. Pass that file as the **5th argument** into `tools/OSV2DRunner.java`.

Inside `tools/OSV2DRunner.java`, the external FaultSeg probability is used as the **voting score input** (`fa`), while orientation scanning still uses a seismic-derived attribute (`faLocal`) from `LocalOrientFilter`. That design is reasonable and consistent with the intent of combining:
- **FaultSeg** for fault likelihood.
- **Seismic structure** for local orientation.

For the 2.5D notebook, the FaultSeg output is also used correctly:
- If a cached full 3D FaultSeg volume exists, each inline/crossline section is taken from that volume.
- Otherwise, the helper `tools/osv_2p5d_volume.py` runs FaultSeg on a local 3D window, extracts the target section fault probability slice, sharpens it, and passes it to `OSV2DRunner`.

### Important limitation
The current repo does **not** provide a notebook that runs the original paper's full **3D OptimalSurfaceVoter** directly on the FaultSeg volume.

The file `OSV_F3_SGY_Demo.ipynb` is therefore not a full 3D OSV implementation. It is effectively another 2D inline demo with optional full-volume FaultSeg caching and SEG-Y export. The only notebook that produces volume-like OSV output today is the **2.5D** notebook, which is a stack of 2D runs, not true 3D optimal surface voting.

### Correctness conclusion
- The **FaultSeg output is taken into account** in the 2D, root demo, and 2.5D workflows.
- The design choice in `tools/OSV2DRunner.java` to use seismic linearity for orientation scanning and FaultSeg probability for voting is technically defensible.
- The main mismatch is naming/expectation: the root demo notebook is not full 3D OSV.
- The 2.5D progress/assembly cells previously contained hardcoded inline-only constants; those were updated so crossline mode and metadata now track the selected axis correctly.

## Parameters You Can Tune

### FaultSeg inference
- `faultSeg_2019_pyTorch/predict_sgy_inline.py`
- `PATCH_N1`, `PATCH_N2`, `PATCH_N3`: inference patch size.
- `OVERLAP`: tile overlap during inference.
- `inline_window`: neighborhood width for inline prediction.
- `chunk_inlines`, `chunk_overlap`: full-volume inference chunking.
- `model_path`: which trained `.pth` weights are used.
- `device`: CPU vs CUDA.

### Seismic preprocessing
- `max_traces` in the 2D/root demos: decimation amount for wide sections.
- Percentile clipping (`p1`, `p99`) before amplitude normalization.

### FaultSeg conditioning before OSV
- `SHARPEN_POWER` in the notebooks and 2.5D helpers.
- Lower values preserve more diffuse probability.
- Higher values suppress background but can break continuity if too aggressive.

### 2D OSV extraction
- `OSV_EXTRACT_THRESHOLD`: threshold applied to `fvtg` to create the binary extraction.
- Lower threshold: more complete but noisier extraction.
- Higher threshold: cleaner but sparser extraction.

### OSV internal behavior
In `tools/OSV2DRunner.java`, the main tuning knobs are:
- `LocalOrientFilter(4.0, 1.0)`: seismic orientation smoothing scale.
- `FaultOrientScanner2(8.0)`: dip scanner smoothing/scale.
- Dip scan ranges in `bestScan(...)`: controls which dips are searched.
- `OptimalPathVoter(15, 30)`: voting window/path geometry.
- `setStrainMax(0.25)`: how much path deformation is allowed.
- `setPathSmoothing(2.0)`: smoothness of the voted paths.
- `prePower`, `preFloor`: pre-voting gating by the guidance attribute.
- `postPower`, `postFloor`: post-voting gating by the guidance attribute.
- Seed threshold candidates in `chooseThreshold(...)`.
- `minSeeds`: minimum acceptable seed count before backing off to weaker thresholds.

### 2.5D assembly behavior
- `OSV_2P5D_AXIS`: `inline` or `crossline`.
- `MAX_2P5D_SECTIONS`: partial run size for tests.
- `section_window` in `run_volume_osv_2p5d_from_sgy(...)`: local 3D context used before extracting the target section.

## Minimal Files To Reuse In Another App

### If your new app already has FaultSeg output and you only want OSV on top
Keep these files:
- `tools/OSV2DRunner.java`
- `src/osv/*.java`
- `build.gradle`
- `libs/*.jar`

Because `build/` is now treated as generated output, run `gradle classes` after cloning before executing the notebooks or any workflow that expects `build/classes` to exist.

You also need code that:
1. Loads your seismic section and FaultSeg probability section.
2. Normalizes seismic amplitudes.
3. Clips FaultSeg probabilities to `[0, 1]` and optionally sharpens them.
4. Writes both arrays to big-endian float32 `.dat` with shape order `[n2][n1]` on disk.
5. Runs `java --class-path libs/*:build/classes tools/OSV2DRunner.java ...`.
6. Reads back `fvg.dat`, `fvtg.dat`, and optionally thresholds `fvtg`.

### If your new app also wants to run FaultSeg inference from SEG-Y
Add these files:
- `faultSeg_2019_pyTorch/predict_sgy_inline.py`
- `faultSeg_2019_pyTorch/unet3_pytorch.py`
- one trained model file under `faultSeg_2019_pyTorch/model/`
- Python dependencies from `requirements-osv.txt`

Optional convenience files:
- `tools/osv_2p5d_volume.py` if you want the current resumable 2.5D section-by-section workflow.
- The three root notebooks if you want the existing interactive demos unchanged.

### If you want true 3D OSV from FaultSeg volume
The current notebooks are not enough by themselves. You would need to build a new driver that:
- loads a 3D FaultSeg probability volume,
- computes or imports 3D orientations,
- calls the Java 3D classes such as `FaultOrientScanner3` and `OptimalSurfaceVoter`,
- and then exports the resulting 3D voting/skinning products.

## Repository Hygiene

### What is local-only and should not go to GitHub
The largest local-only content in this workspace is:
- `.venv/`: local Python environment.
- `notebooks/`: runtime outputs, checkpointed arrays, generated SEG-Ys, previews, and executed notebooks.
- `faultSeg_2019_pyTorch/model/`: trained weights.
- `faultSeg_2019_pyTorch/output/`: training logs, checkpoints, and figures.
- `faultSeg_2019_pyTorch/data/`: local training/validation arrays.
- `data/F3/`: the local SEG-Y input data.
- `referenceDocumentation/*.pdf`: local copies of papers.
- `build/`, `*.class`, `.DS_Store`, and scratch files.

The ignore rules in `.gitignore` were updated accordingly.
