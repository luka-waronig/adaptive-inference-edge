# Reproducing the experiments

These instructions follow the two notebooks in this repository. The dataset and trained checkpoints are intentionally excluded from Git. Full training can take a substantial amount of time; a smoke run checks the pipeline but does **not** reproduce the reported metrics.

## 1. Environment

Use Python 3.12 or 3.13. From the repository root, install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

The saved baseline environment in the notebook output used Python 3.13 with torch 2.10.0, torchvision 0.25.0, timm 1.0.21, fvcore 0.1.5.post20221221, numpy 2.5.3, pandas 2.3.3, and matplotlib 3.11.2. Hardware-specific details are recorded by the notebooks; other compatible environments may produce small numerical differences.

## 2. Dataset setup

1. Download NWPU-RESISC45 from the dataset authors or the archive linked in `README.md`.
2. Extract the class folders to:

   ```text
   data/NWPU-RESISC45/
   ```

   The directory should contain 45 class subdirectories with 700 images each.
3. Run the notebooks from the repository root. The notebooks download the three small published train/validation/test split manifests when absent and verify their SHA-256 hashes. The image archive itself is not downloaded or extracted automatically.

The split protocol yields 18,900 train images, 3,150 `val_select`, 3,150 `val_policy`, and 6,300 test images. The published validation partition is divided into `val_select` and `val_policy` by the notebook code.

## 3. Run the static baseline first

Open `I_static_vit_baseline.ipynb` and run it from the repository root.

- Use `RUN_MODE = "smoke"` only for a quick end-to-end pipeline check.
- Use `RUN_MODE = "full"` to train the reported 30-epoch baseline.
- The notebook uses DeiT-Tiny (`deit_tiny_patch16_224.fb_in1k`), 224 × 224 model inputs, batch size 32, seed 42, and split seed 2026 in the recorded setup.
- The pretrained weights are fetched on first use unless a local pretrained-weight file is configured.

The full baseline run should create artifacts under:

```text
runs/baseline_deit_tiny_seed42/
├── best.pt
├── last.pt
├── baseline_summary.json
├── run_config.json
├── environment.json
├── split_manifest.csv
├── class_to_idx.json
├── flops.json
└── figures/
```

Exact artifact lists can vary slightly with notebook revisions. The adaptive notebook requires `best.pt`, `baseline_summary.json`, `run_config.json`, `split_manifest.csv`, `class_to_idx.json`, and `flops.json` from this directory. Preserve these artifacts together; the adaptive notebook verifies the baseline checkpoint identity against the saved summary.

`best.pt` is the selected model checkpoint. `last.pt` is the resumable training state and is not the file to substitute for the selected checkpoint unless the notebook explicitly calls for it.

## 4. Run the adaptive notebook

After the baseline full run completes, open `II_adaptive_vit_inference.ipynb` and run it from the same repository root.

- Set `RUN_MODE = "full"` for the reported experiment.
- It loads the baseline `best.pt` as the frozen teacher/reference and initializes a separate adaptive model.
- The recorded setup uses 15 adaptive training epochs, batch size 32, seed 42, exit blocks 4 / 8 / 12, distillation temperature 3.0, auxiliary loss weight 0.5, distillation loss weight 0.5, and a maximum allowed validation accuracy loss of 1.0 percentage point for policy selection.
- It partitions `val_policy` into a calibration subset (1,565 images) and a policy-selection subset (1,585 images). Temperatures are fitted on the calibration subset; thresholds are selected on the separate policy-selection subset.
- The held-out test set is evaluated after the checkpoint, temperatures, thresholds, and resource policy have been selected.

The adaptive run writes to `runs/adaptive_deit_tiny_seed42/`, including the selected `best.pt`, resumable `last.pt`, configuration and environment metadata, training history, FLOPs and policy metadata, validation sweep, test summary, latency measurements, resource trace, and figures.

## 5. Google Colab notes

If running on Colab, upload or clone the repository so both notebooks and `requirements.txt` are in the same project root. Upload or extract the dataset into `data/NWPU-RESISC45/`. Keep `runs/` on persistent storage (for example, mounted Google Drive) if you need the baseline artifacts to survive runtime resets.

The baseline checkpoint is not enough by itself: the adaptive notebook also requires the baseline summary, run configuration, split manifest, class mapping, and FLOPs metadata. Copy the complete `runs/baseline_deit_tiny_seed42/` artifact directory to persistent storage, not just `best.pt`. Restore that directory in the adaptive notebook before running it.

Do not change run settings while reusing an existing run directory. The notebook checks configuration identity and may stop if the saved checkpoint was produced with different settings. Use a new run name for a genuinely different experiment.

## 6. Verify outputs

Compare the new outputs with the reported results from the recorded experiments. Small differences can arise from runtime, device, and dependency changes. 

Important checks:

- Baseline best validation checkpoint and its metadata are present.
- Adaptive notebook loads the exact saved baseline checkpoint.
- Test set size is 6,300.
- Static and adaptive FLOPs use the same `fvcore_fma1_scalar_v1` counting convention.
- Threshold selection uses policy-validation data, not the test set.
- Latency comparison uses actual conditional inference, batch one, CPU FP32, one thread, and 256 sampled test images.

## 7. Git hygiene

The repository's `.gitignore` excludes `data/` and `runs/`. Keep large datasets and checkpoints out of ordinary Git commits. Commit notebooks, requirements, documentation, and selected small figures. If distributing checkpoints is necessary, use an appropriate artifact store or Git LFS and document the exact version/hash.
