# Experiment Configs

The provided configs target the standard MVTec AD category layout under
`mvtec_datasets/` and write results under `results/mvtec/`.

`configs/mvtec/baselines/` contains the two reference settings:

- `mvtec_streaming_mh_patchcore.yaml`
- `mvtec_vanilla_patchcore.yaml`

`configs/mvtec/ablations/` contains the MVTec ablations used to vary the memory
budget, chunk size, clustering strategy, Mahalanobis normalization, and GeoReS
selection.

All shipped configs set `execution.device: cpu`.
