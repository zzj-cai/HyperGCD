# HSI-GCD Base: Model and Training

This is a standalone, training-only release of the non-distilled Base model.
It contains the HyperSIGMA-Base spatial/spectral encoders, feature fusion,
known-class memory prototypes, and the training dependencies. No files from
the original private project are required at runtime.

Distillation, ablations, clustering evaluation, visualization, batch runners,
datasets, and pretrained weights are intentionally not included. This partial
release does not provide the complete paper evaluation pipeline or reproduce
reported accuracies on its own.

## Files

| File | Purpose |
| --- | --- |
| `models/spatial.py`, `models/spectral.py` | HyperSIGMA-Base encoders |
| `models/fusion.py` | Spatial-spectral fusion and contrastive projector |
| `models/prototype.py` | EMA memory prototypes for known classes only |
| `train.py` | Single-dataset training entry point |
| `data.py` | Patch loading, labeled-known split, and SLIC superpixels |
| `sampler.py` | Same-superpixel positives and dissimilarity-weighted negatives |
| `losses.py` | Class-supervised and superpixel-supervised contrastive loss |
| `optimizer.py` | AdamW parameter grouping used by the Base experiment |

## Environment

Use Python 3.8 or newer. Install a matching PyTorch/torchvision pair for your
CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The dependency ranges are compatibility constraints, not a fully pinned or
GPU-tested environment. Install compatible versions for your Python and CUDA
versions. AMP is enabled on CUDA. The default batch requires substantial GPU
memory; `--batch-size` must be a multiple of three and at least six.

## Data and Pretrained Weights

Place the following MAT files directly in the directory passed to `--data-root`.
The image must have shape `(height, width, bands)` and the label map must have
shape `(height, width)`, with zero denoting background.

| Dataset argument | Image file / MAT key | Label file / MAT key | Known class IDs (original) |
| --- | --- | --- | --- |
| `Houston` | `Houston13.mat` / `HSI` | `Houston13_gt.mat` / `gt` | 1, 2, 3, 4, 5, 11, 12, 13, 14, 15 |
| `Pavia` | `Pavia.mat` / `paviaU` | `Pavia_gt.mat` / `Data_gt` | 1, 2, 3, 4, 5, 6 |
| `Trento` | `Trento-HSI.mat` / `HSI` | `Trento-GT.mat` / `GT` | 1, 2, 3 |

Supply the official HyperSIGMA-Base spatial and spectral pretrained checkpoints
through `--spat-checkpoint` and `--spec-checkpoint`. A checkpoint can contain a
`model` dictionary or be a plain encoder state dictionary. As in the source Base
experiment, only matching names and shapes are loaded; mismatched embeddings
are not interpolated. Load trusted checkpoint files only.

## Training

Run from this directory. Replace the data and checkpoint paths with your own:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --dataset Houston \
  --data-root /path/to/data \
  --spat-checkpoint /path/to/spatial_base.pth \
  --spec-checkpoint /path/to/spectral_base.pth \
  --device cuda:0 \
  --seed 42 \
  --output-dir checkpoints/Houston_base_seed42
```

Use `--dataset Pavia` or `--dataset Trento` for the other datasets. For a two-epoch
smoke test, append `--epochs 2 --checkpoint-every 1 --num-workers 0` and use a
fresh output directory. Smaller batches can reduce memory use but change the
contrastive objective and need not reproduce full-batch results.

| Parameter | Default |
| --- | --- |
| Epochs | 100, fixed duration; no evaluation-based early stopping |
| Learning rate | 0.0001, cosine decay to 0.000001 |
| Batch size | 1536 patches arranged as anchor/positive/negative triples |
| Gradient accumulation | 4 batches per optimizer step |
| Patch size | 9 |
| Labeled ratio | 0.10 within each known class |
| Supervised weight | 0.35 |
| Unsupervised weight | 0.65 |
| Contrastive/prototype temperature | 0.10 |
| Prototype EMA momentum | 0.90 |
| Requested SLIC segments | `max(100, foreground_pixels // 40)` |

Pass `--sp-segments 300` to request a fixed count instead. SLIC's actual count
can differ from the requested count. All spectral channels enter the network;
three-component PCA is used only to construct superpixels. The two input views
are identical without augmentation; stochastic depth remains part of the model.

The objective is:

```text
L = lambda * (L_CE + L_SupCon) + (1 - lambda) * L_SPCon
```

`L_CE` and `L_SupCon` use only labeled-known samples. `L_SPCon` uses superpixel
IDs on unlabeled samples, which include both unlabeled known and novel pixels.
Novel semantic labels never enter the loss. The foreground mask is used to
select the transductive training population. A positive pair shares a superpixel;
negative-superpixel probability is proportional to clipped spectral-center
dissimilarity `1 - cosine`. This is a sampling score, not an entropy-based
uncertainty loss. Prototype memory represents known classes only. The sampler
produces triples, but no triplet loss is used because its source weight is zero.

## Outputs and Scope

Each run writes `run_config.json` and `checkpoint_epN.pt` to `--output-dir`.
The directory must be empty to prevent accidental overwrites. Checkpoints
contain `state_dict`, `epoch`, and `config`; model keys start with `backbone.`
or `prototype_head.`. They store model weights, not optimizer/resume state.
The final fixed-epoch checkpoint is always saved. There is no accuracy-selected
"best" checkpoint, accuracy report, or downstream clustering in this release.

The model/loss/sampling choices follow the source Base experiment. The optimizer
uses a plain PyTorch AdamW with the original parameter grouping so AMP does not
depend on an MMEngine optimizer wrapper. Sampling probabilities are normalized
in float64, with an explicit fallback for degenerate dissimilarities.
Empty superpixels are excluded from the negative distribution, which is the
conditional distribution of the source sampler's accepted negatives.
Independent superpixel caches do not reuse private-project caches. These packaging changes
and differing runtime versions do not guarantee bitwise-identical runs.

## Attribution

The encoder implementations are adapted from HyperSIGMA and retain the BEiT
copyright/license headers present in the original files, including the upstream
implementation references. Dataset and pretrained checkpoint redistribution
rights must be checked separately; neither is bundled here. This package does
not assign a new license to third-party code or pretrained weights. The MIT
notice carried by the BEiT-derived encoders is included in
`THIRD_PARTY_LICENSES.txt`. A license for the project's own contributions has
not been selected; that decision remains with the copyright holders before
public distribution.
