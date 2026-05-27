# Mahalanobis PatchCore

Mahalanobis PatchCore is a compact PatchCore-style anomaly detector for visual
inspection. It keeps the nearest-neighbor scoring structure of PatchCore and
adds a streaming-compatible Mahalanobis feature normalization stage before the
memory bank is built.

The repository contains the runtime implementation and the MVTec AD experiment
configs used for the baseline and ablation runs. The default experiments run on
CPU and expect the MVTec AD categories under `mvtec_datasets/`.

## Layout

- `mhpc/`: runtime package and active plugin implementations.
- `configs/mvtec/baselines/`: canonical MVTec baseline configs.
- `configs/mvtec/ablations/`: MVTec ablation configs.
- `scripts/run_all_experiments.sh`: batch runner for the provided configs.
- `tests/`: focused runtime and numerical checks.

## Setup

Create an environment with Python 3.11 and install the pinned dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The runtime uses `faiss-cpu` and CPU Torch wheels by default. GPU execution is
not required for the provided configs.

## Dataset

Place MVTec AD in a directory named `mvtec_datasets` at the repository root:

```text
mvtec_datasets/
  bottle/
  cable/
  ...
  zipper/
```

Each category is expected to use the standard MVTec AD train/test/ground-truth
layout.

## Running An Experiment

Run a single config with:

```bash
python run_mhpc.py --config configs/mvtec/baselines/mvtec_streaming_mh_patchcore.yaml
```

Run all provided MVTec configs with:

```bash
bash scripts/run_all_experiments.sh --all --continue
```

Results are written under `results/mvtec/<experiment>/<timestamp>/`.

## Exporting Teacher Artifacts

TinyEdge student training needs a fitted PatchCore teacher, its memory bank,
and frozen replay features from the normal training split. Generate those files
with the teacher artifact config:

```bash
python run_mhpc.py --config configs/mvtec/teacher/mvtec_streaming_mh_patchcore_teacher_artifacts.yaml
```

The export uses the same MVTec layout described above and writes everything
under the run directory:

```text
results/mvtec/mvtec_streaming_mh_patchcore_teacher_artifacts/<timestamp>/artifacts/
  bottle/
    model/
      checkpoint/
        manifest.json
        ...
      memory_bank/
        memory_bank.h5
    train/
      transform/
      distance/
      scoring/
    test/
      transform/
      distance/
      scoring/
```

The `model/checkpoint` folder stores the fitted teacher checkpoint and a small
manifest. `model/memory_bank/memory_bank.h5` stores the fitted reference bank.
The `train` and `test` replay folders store selected slot-boundary payloads
grouped by sample group, with matching metadata JSONL files.

These payloads are generated runtime artifacts. Keep them local or publish them
through your own release process; they are not tracked in this repository.

## Tests

Run the focused runtime suite with:

```bash
pytest -q
```
