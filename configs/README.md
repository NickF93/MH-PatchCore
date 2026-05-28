# Experiment Configs

The provided configs target the standard MVTec AD category layout under
`mvtec_datasets/` and write results under `results/mvtec/`.

`configs/mvtec/baselines/` contains the two reference settings:

- `mvtec_streaming_mh_patchcore.yaml`
- `mvtec_vanilla_patchcore.yaml`

`configs/mvtec/teacher/` contains the export setting used to produce fitted
teacher artifacts for downstream student training:

- `mvtec_streaming_mh_patchcore_teacher_artifacts.yaml`

`configs/mvtec/ablations/` contains the MVTec ablations used to vary the memory
budget, chunk size, clustering strategy, Mahalanobis normalization, and GeoReS
selection.

All shipped configs set `runtime.device: cpu`.

## Teacher Artifact Export

The `teacher_export` block is optional. When omitted, no teacher checkpoint,
memory-bank payload, or replay payload is written.

```yaml
teacher_export:
  checkpoint:
    enabled: true
  memory_bank:
    enabled: true
  replay:
    enabled: true
    slots:
      - transform
      - distance
      - scoring
```

`replay.slots` accepts inference slots only: `dataloader`, `backbone`,
`patch_align`, `preprocess`, `feature_agg`, `proj1`, `transform`, `proj2`,
`distance`, and `scoring`. The state-building stages `mem_agg` and
`materialize` are represented by the checkpoint and memory-bank artifacts
instead of replay files.

Replay files are written batch-major: each HDF5 row represents one source
image. Patch-level payloads are flattened inside that image row before the file
is split by sample group.

Run the teacher export config with:

```bash
python run_mhpc.py --config configs/mvtec/teacher/mvtec_streaming_mh_patchcore_teacher_artifacts.yaml
```

The generated files live under:

```text
results/mvtec/mvtec_streaming_mh_patchcore_teacher_artifacts/<timestamp>/artifacts/<dataset>/
```

Use that `artifacts` directory as the teacher-artifact root for downstream
student training tools.
