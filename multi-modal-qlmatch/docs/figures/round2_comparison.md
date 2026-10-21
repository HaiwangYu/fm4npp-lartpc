# Round-2 comparison (R2P protocol, test split)

Frozen backbone + head at 0.1 / 1 / 10 % labels; fine-tune at 0.1 / 1 %; late-fused light for every backbone; 'C-only diag' = frozen at 1 % without late light.


## evt_nu: event ν/cosmic AUC

| method | frozen 0.1% | frozen 1% | frozen 10% | FT 0.1% | FT 1% | C-only diag (frozen 1%) |
|---|---|---|---|---|---|---|
| qlfm-M CL (MAE, 21M) | 0.986±0.002 | 0.999 | 0.999 | 0.985 | 0.998 | 0.999 |
| qlfm-S CL (MAE, 6.6M) | 0.953±0.001 | 0.970 | 0.974 | 0.953 | 0.990 | 0.972 |
| qlfm-S C (MAE, 6.6M) | 0.943±0.001 | 0.960 | 0.962 | 0.664 | 0.956 | 0.666 |
| qlfm-SD CL (self-distill., 21M) | 0.949±0.000 | 0.959 | 0.959 | 0.940 | 0.959 | 0.605 |
| qlfm-SD C (self-distill., 21M) | 0.942±0.000 | 0.957 | 0.957 | 0.690 | 0.952 | 0.609 |
| Panda (PTv3+Sonata, 21M) | 0.938±0.007 | 0.978 | 0.984 | 0.830 | 0.968 | 0.914 |
| PoLAr-MAE (ViT-S, 22M) | 0.950±0.001 | 0.962 | 0.963 | 0.920 | 0.964 | 0.660 |
| FM4NPP (Mamba2, 5M, per hit) | 0.943±0.000 | 0.961 | 0.963 | 0.626 | 0.944 | 0.712 |
| Pandora+BDT C (no NN) | 0.926±0.001 | 0.949±0.000 | 0.955 |  |  |  |
| Pandora+BDT CL (no NN) | 0.957±0.003 | 0.976±0.000 | 0.983 |  |  |  |
| Pandora direct (no training) | 0.826 | | | | | |

## flavor: flavor macro-F1

| method | frozen 0.1% | frozen 1% | frozen 10% | FT 0.1% | FT 1% | C-only diag (frozen 1%) |
|---|---|---|---|---|---|---|
| qlfm-M CL (MAE, 21M) | 0.531±0.003 | 0.713 | 0.755 | 0.640 | 0.724 | 0.709 |
| qlfm-S CL (MAE, 6.6M) | 0.416±0.002 | 0.624 | 0.642 | 0.512 | 0.622 | 0.625 |
| qlfm-S C (MAE, 6.6M) | 0.457±0.000 | 0.509 | 0.520 | 0.389 | 0.469 | 0.429 |
| qlfm-SD CL (self-distill., 21M) | 0.389±0.003 | 0.450 | 0.423 | 0.316 | 0.382 | 0.297 |
| qlfm-SD C (self-distill., 21M) | 0.369±0.001 | 0.437 | 0.439 | 0.314 | 0.274 | 0.303 |
| Panda (PTv3+Sonata, 21M) | 0.465±0.001 | 0.679 | 0.734 | 0.506 | 0.690 | 0.666 |
| PoLAr-MAE (ViT-S, 22M) | 0.446±0.004 | 0.554 | 0.570 | 0.411 | 0.563 | 0.497 |
| FM4NPP (Mamba2, 5M, per hit) | 0.445±0.003 | 0.546 | 0.557 | 0.355 | 0.485 | 0.507 |
| Pandora+BDT C (no NN) | 0.542±0.007 | 0.619±0.001 | 0.641 |  |  |  |
| Pandora+BDT CL (no NN) | 0.559±0.007 | 0.632±0.001 | 0.655 |  |  |  |
| Pandora direct (no training) | 0.547 | | | | | |

## energy: E resolution (lower better)

| method | frozen 0.1% | frozen 1% | frozen 10% | FT 0.1% | FT 1% | C-only diag (frozen 1%) |
|---|---|---|---|---|---|---|
| qlfm-M CL (MAE, 21M) | 0.418±0.003 | 0.284 | 0.276 | 0.385 | 0.303 | 0.284 |
| qlfm-S CL (MAE, 6.6M) | 0.465±0.007 | 0.354 | 0.333 | 0.411 | 0.318 | 0.345 |
| qlfm-S C (MAE, 6.6M) | 0.484±0.001 | 0.378 | 0.378 | 0.563 | 0.412 | 0.441 |
| qlfm-SD CL (self-distill., 21M) | 0.483±0.005 | 0.367 | 0.370 | 0.469 | 0.415 | 0.439 |
| qlfm-SD C (self-distill., 21M) | 0.475±0.002 | 0.391 | 0.387 | 0.477 | 0.426 | 0.454 |
| Panda (PTv3+Sonata, 21M) | 0.496±0.001 | 0.343 | 0.302 | 0.484 | 0.358 | 0.423 |
| PoLAr-MAE (ViT-S, 22M) | 0.469±0.002 | 0.389 | 0.381 | 0.566 | 0.379 | 0.421 |
| FM4NPP (Mamba2, 5M, per hit) | 0.454±0.013 | 0.369 | 0.358 | 0.567 | 0.334 | 0.420 |
| Pandora+BDT C (no NN) | 0.407±0.005 | 0.355±0.003 | 0.324 |  |  |  |
| Pandora+BDT CL (no NN) | 0.400±0.005 | 0.342±0.001 | 0.308 |  |  |  |
| Pandora direct (no training) | 0.428 | | | | | |

## hit: hit charge-wtd F1

| method | frozen 0.1% | frozen 1% | frozen 10% | FT 0.1% | FT 1% | C-only diag (frozen 1%) |
|---|---|---|---|---|---|---|
| qlfm-M CL (MAE, 21M) | 0.868±0.000 | 0.892 | 0.894 | 0.905 | 0.947 | 0.891 |
| qlfm-S CL (MAE, 6.6M) | 0.502±0.005 | 0.685 | 0.685 | 0.783 | 0.902 | 0.675 |
| qlfm-S C (MAE, 6.6M) | 0.080±0.001 | 0.429 | 0.440 | 0.008 | 0.828 | 0.170 |
| qlfm-SD CL (self-distill., 21M) | 0.010±0.001 | 0.141 | 0.116 | 0.001 | 0.818 | 0.000 |
| qlfm-SD C (self-distill., 21M) | 0.032±0.003 | 0.147 | 0.122 | 0.001 | 0.657 | 0.004 |
| Panda (PTv3+Sonata, 21M) | 0.536±0.003 | 0.715 | 0.725 | 0.656 | 0.823 | 0.629 |
| PoLAr-MAE (ViT-S, 22M) | 0.167±0.025 | 0.537 | 0.526 | 0.491 | 0.670 | 0.351 |
| FM4NPP (Mamba2, 5M, per hit) | 0.200±0.013 | 0.463 | 0.468 | 0.371 | 0.620 | 0.238 |
| Pandora ν-slice hits (no training) | see round 1: 0.63–0.65 | | | | | |
