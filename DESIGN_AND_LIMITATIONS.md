# Design choices and limitations

## Problem setting

The project considers a resource-constrained field gateway that classifies independent remote-sensing image tiles for local indexing. NWPU-RESISC45 is a benchmark used to evaluate the idea; it is not a recorded operational stream. The project does not perform temporal change detection.

## Model design

### Static reference

The reference model is `deit_tiny_patch16_224.fb_in1k`, fine-tuned for 45 scene classes. It processes each input through all 12 transformer blocks. Its checkpoint is frozen before adaptive training and remains the reference for evaluation and knowledge distillation.

### Shared-weight early exits

The adaptive model uses a shared transformer trunk with classification exits after blocks 4, 8, and 12. Intermediate exits reuse the same representations and trunk weights; this is not an ensemble of three independent networks. At inference, a prediction may return at block 4, continue to block 8, or reach block 12. A return from an intermediate exit skips later blocks.

### Distillation and supervision

The adaptive training objective combines supervised cross-entropy for the final and intermediate predictions with a KL-divergence distillation term from the frozen full-depth teacher. The recorded settings use distillation temperature 3.0 and weights 0.5 for auxiliary exit supervision and 0.5 for distillation. These are the selected implementation settings, not a claim that a broad hyperparameter search was performed.

### Calibration and policy selection

The `val_policy` partition is split into disjoint calibration and policy-selection subsets by class. Temperature scaling is fit on the 1,565-image calibration subset. Thresholds are then selected on the 1,585-image policy-selection subset, with a target allowing at most a 1.00 percentage-point accuracy loss relative to the frozen teacher on that subset. The recorded thresholds are 0.90 for exit 4 and 0.50 for exit 8.

The 6,300-image test partition is held out until the checkpoint and policy are fixed. Test-set ECE values are reported as post-selection diagnostics and are not used to choose temperatures or thresholds.

## Compute and latency methodology

FLOPs are counted with `fvcore` under the shared `fvcore_fma1_scalar_v1` convention, with one multiply-add counted as one FLOP. The code includes the exit/gating computation in path profiling and checks the static and adaptive full-depth cost for consistency.

The common latency comparison measures model-forward inference on CPU using FP32, batch size one, one thread, and 256 sampled test images. It is not an embedded-device benchmark. Latency depends on the hardware, runtime, thread settings, and measurement conditions.

The resource-response experiment replays a simulated sequence of changing maximum-depth allowances across test examples. It demonstrates that the controller respects the imposed depth cap under the tested trace; it is not evidence of operation on a live sensor stream.

## What the results support

For the recorded NWPU-RESISC45 experiment and DeiT-Tiny implementation:

- Confidence-guided early exiting reduces mean FLOPs compared with always using all 12 blocks.
- The selected adaptive policy retains most of the static model's test accuracy while lowering average computation.
- At matched exit counts and mean FLOPs, confidence-guided allocation performs better than random allocation in the recorded control experiment.
- The measured CPU benchmark shows lower median latency for the selected policy under the tested conditions.

## Limitations

1. **Dataset scope:** results are on one balanced remote-sensing scene-classification benchmark, not a broad evaluation across sensors, regions, seasons, or domain shifts.
2. **Model scope:** only one DeiT-Tiny backbone family is evaluated. Results do not establish superiority over all CNNs, transformers, or compression methods.
3. **Hardware scope:** latency is measured on a CPU, not a Jetson device, mobile NPU, or production edge gateway.
4. **Energy:** FLOPs and latency are not direct measurements of energy in joules, thermal behaviour, or battery life.
5. **Streaming behaviour:** the resource schedule is simulated and the images are independent; no real-time stream, temporal correlation, or end-to-end ingestion overhead is evaluated.
6. **Accuracy trade-off:** the adaptive model's full-depth test accuracy (94.21%) is below the frozen static baseline's 94.71%, indicating that adaptive training changes the full-depth model as well as enabling early exits.
7. **Reproduction variance:** recorded results are notebook outputs from a particular software/hardware environment. Clean reruns may vary slightly, and the documentation update itself did not rerun training or evaluation.
8. **Calibration under shift:** temperature scaling and thresholds are selected on the benchmark validation data. Calibration drift under distribution shift is not evaluated.
9. **Search scope:** training settings and threshold candidates are explicitly configured; the experiment is not an exhaustive search over architectures, hyperparameters, or all possible stopping policies.

## Next experiments if more compute/time becomes available

- Measure latency, energy, and thermal behaviour on a target edge device.
- Evaluate calibration and accuracy under geographic or sensor domain shift.
- Compare against additional compact CNN and transformer baselines.
- Explore exit-aware distillation or structured dropout to reduce the full-depth accuracy gap.
- Test threshold adaptation under changing compute budgets and uncertainty distributions.
