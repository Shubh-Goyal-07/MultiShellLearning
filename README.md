# ShellMetric

ShellMetric is a prototype-free supervised metric-learning implementation with
learned ordered radii, deterministic confusion-driven shell plans, and
decoder-free evaluation. The normative design is
[`FINAL_SHELLMETRIC_IMPLEMENTATION_SPEC.md`](FINAL_SHELLMETRIC_IMPLEMENTATION_SPEC.md).

## Install

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[dev,reporting]"
python -m pytest -q
```

The checked-in `environment.yml` provides a CUDA-enabled Conda environment.

## Run a study

One restartable command expands a suite, prints every reporting row and every
deduplicated encoder job, and then trains and evaluates them:

```powershell
python scripts/run_shellmetric_study.py --config configs/shellmetric/mnist_smoke.yaml --dry-run
python scripts/run_shellmetric_study.py --config configs/shellmetric/mnist_smoke.yaml
python scripts/run_shellmetric_study.py --config configs/shellmetric/mnist_smoke.yaml --resume
```

`--fixed-shell-counts compact|all|none|1,2,4` overrides the FixedS sweep, and
`--device auto|cpu|cuda` selects the device (default `auto`). An interrupted job
continues only with `--resume`. A finished job is always reused.

Planning is cached by content hash: split manifest, cross-fitted CE pilot, raw
soft confusion, and AutoK. Encoder jobs live in `cache_dir/jobs/<job hash>`. The
hash covers only training semantics: plan semantic hash, data split, backbone,
activation, output map, dimension, loss, P×K schedule, optimizer, budget, and
seed. Suites that share a `cache_dir` therefore share jobs. AutoK/FixedS
aliases, the Stage-A ReLU entry, and the d = 3/32/128 cells of the dimension
curve each train once. Each study writes its rows to
`<output_dir>/results/<method>/d<d>/seed<s>/`.

The official test partition stays sealed. Test evaluation requires both
`evaluation.locked: true` and `--allow-test`, and is refused for Stage-A/B
architecture candidates.

## Suites

| Config | Protocol stage |
|---|---|
| `mnist_smoke.yaml`, `fashion_mnist_smoke.yaml` | pipeline smoke (small CNN, d=3, one seed) |
| `cifar100_shellmetric.yaml` | canonical shell study and the three controls |
| `cifar10_shell_study.yaml` | canonical shell-count study on CIFAR-10 |
| `cifar{10,100}_core_benchmark.yaml` | AutoK and all eight baselines at d ∈ {3, 32, 128} |
| `cifar100_dimension_curve.yaml` | d ∈ {2, 3, 8, 16, 32, 128, 512, 1024} |
| `tiny_imagenet_benchmark.yaml`, `cub200_benchmark.yaml` | dataset generalization at d=128 |
| `cifar100_resnet50_backbone.yaml` | scratch ResNet-50 confirmation |
| `cifar100_vit_backbone.yaml` | ViT-S/16 confirmation (gated: `blocked_external_pin`) |
| `architecture/*.yaml` | Section 5.4 Stage A, Stage B, and auxiliary result |
| `mnist_rehearsal_primary.yaml`, `architecture/mnist_*.yaml` | cheap rehearsal of the whole workflow (ResNet-18 on MNIST, 15 epochs) |
| `mnist_dimension_curve.yaml` | every method of the MNIST rehearsal (FixedS, AutoK, controls, eight baselines) at d ∈ {2, 3, 32, 128, 512, 1024}; reuses its d=3 jobs |

HyperSpaceX rows report `blocked_external_pin` until `external_pins.hyperspacex`
records an immutable commit, environment, command, and checkpoint rule.
HyperSpaceX-Matched also stays blocked until the official reproduction hash
exists. HyperSpaceX-Official is a reference row and never enters paired
statistics.

### Architecture study (Section 5.4)

1. Run `cifar100_shellmetric.yaml` and lock its loss. Its `provenance.json`
   holds `locked_loss_hash`, `autok_plan_semantic_hash`, and
   `autok_plan_provenance_hash`, which Stage A references.
2. Stage A writes `architecture_decision.json`; Stage B references its literal
   winner and `decision_hash`.
3. Stage B's decision completes `architecture/cifar100_auxiliary_result.yaml`.
   Only this configuration may add seeds 3 and 4 or be tested. If it equals
   ReLU + `linear_no_bias`, its rows alias the primary AutoK run.

`scripts/fill_architecture_config.py` fills a template's `REPLACE_WITH_*`
placeholders from those files and writes `<template>.filled.yaml` next to it:

```powershell
python scripts/fill_architecture_config.py configs/shellmetric/architecture/cifar100_stage_b_output_map.yaml --primary artifacts/cifar100/cifar100_shellmetric --stage-a artifacts/cifar100/cifar100_stage_a_activation
```

Each stage verifies the locked references and prior decisions before it trains.
Stages and controls run only in the CIFAR-100/ResNet-18/3D cell unless
`study.rehearsal: true`, which the MNIST rehearsal configs set; rehearsal
results exercise the workflow and are never publication results.
Configs that request an activation × output-map Cartesian product are rejected.

## Summaries

```powershell
python scripts/summarize_runs.py artifacts/cifar100/cifar100_shellmetric/results --output-dir reports/cifar100
```

This writes separate tables:

- `table_representation.md`: raw Euclidean kNN and retrieval.
- `table_common_probe.md`: the affine probe on the frozen encoder.
- `table_native.md`: ShellMetric's shell-then-cosine kNN and baseline native heads.
- `table_reference.md`: external reproductions.

Rows are grouped by dataset, partition, method, and embedding dimension, so
validation and test results are never pooled. It also writes long and wide CSVs,
and `--reference-method` adds paired confidence intervals that match seeds within
each (dataset, partition, dimension) cell; reference-table rows never enter them.
Failed jobs are counted, never dropped.

## Embedding plots

```powershell
python scripts/visualize_embeddings.py artifacts/mnist/mnist_rehearsal_primary
```

For every evaluated row (method × d × seed × partition, ShellMetric and every
baseline alike) this writes `<study>/visualizations/<partition>/d<d>/seed<s>/`:

- `<method>_3d.html`: interactive 3D scatter coloured by class; ShellMetric rows
  show their learned shells as spheres and each class's assigned shell.
- `<method>_2d.png`: a 2D view plus each class's true ‖z‖ against the learned radii.
- `overview.png`: every method side by side; `visualizations/index.html` links all.

Reduction is for display only. d ≤ 3 is drawn in native coordinates. For d > 3
the default files use PCA about the origin (norms and shells stay readable), and
`--reducers` adds nonlinear views: `umap` (default, needs `umap-learn`) and
`tsne` (needs `scikit-learn`), written as `<method>_3d_umap.html`,
`<method>_2d_umap.png`, `overview_umap.png`, and likewise for `tsne`. They show
neighbourhood structure but not radii; the radius strip always uses the true
d-dimensional norms. A missing package only skips its views, with a warning.

Add `--offline` to embed plotly.js in each HTML file (so it opens without
internet), `--method`/`--seed`/`--partition` to plot a subset, `--max-points` to
trade detail for speed, and `--output DIR` to write a filtered comparison
elsewhere. The index always links every figure already present in the output
directory.

## Implementation map

- `src/multishell/confusion.py`: raw out-of-fold soft confusion `Q`, `W`, `Ŵ`, `h`.
- `src/multishell/train_pilot.py`: cross-fitted CE pilot with the Section 8.4 schedule.
- `src/multishell/training.py`: optimizer, warmup-cosine, AMP, checkpoint, and embedding helpers.
- `src/multishell/shellmetric/plan.py`: `Smax`, `s²` capacities, assignments, plan hashes.
- `src/multishell/shellmetric/autok.py`: paired class-stratified bootstrap and the one-SE rule.
- `src/multishell/shellmetric/radii.py`, `loss.py`, `sampler.py`: ordered RMS radii,
  pair/shell losses, and per-epoch paired P×K schedules.
- `src/multishell/shellmetric/train.py`: decoder-free training with validation-1NN selection.
- `src/multishell/shellmetric/evaluate.py`: exact blockwise kNN, retrieval,
  shell-then-cosine kNN, and geometry diagnostics.
- `src/multishell/shellmetric/probe.py`: the frozen-encoder affine probe.
- `src/multishell/train_baseline.py`, `baselines.py`: the eight matched baselines.
- `src/multishell/shellmetric/study.py`: row and job expansion, deduplication,
  controls, dimension suites, and external gates.
- `src/multishell/shellmetric/architecture.py`: Stage A/B validation and the traced decision rule.
- `src/multishell/visualize.py`: 3D/2D embedding-space figures for every row.
- `src/multishell/shellmetric/stages.py`, `jobs.py`, `runner.py`: cached
  planning, the semantic job store, and the CLI.

Run `python -m ruff check src tests scripts` and `python -m pytest -q` before a
study release.
