# EDM QM9 reproduction report

Date: 2026-07-20

Working directory:

```text
/g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM
```

Repository:

```text
https://github.com/ehoogeboom/e3_diffusion_for_molecules.git
commit fce07d701a2d2340f3522df588832c2c0f7e044a
```

Paper downloaded to:

```text
EDM/edm_paper_2203.17003.pdf
```

Paper:

```text
Hoogeboom et al., Equivariant Diffusion for Molecule Generation in 3D, ICML 2022
arXiv:2203.17003
```

## 1. What the paper does

EDM models a molecule as atom coordinates plus atom-type features:

$$
x \in \mathbb{R}^{M \times 3}, \qquad h \in \mathbb{R}^{M \times d}.
$$

The diffusion model jointly noising and denoising:

- continuous coordinates $x$,
- categorical atom-type features $h$,
- optional integer charge features.

The denoiser is an EGNN, so the reverse diffusion process is $E(3)$ equivariant. Coordinates are modeled in the zero-center-of-gravity subspace, which handles translation invariance correctly. Atom ordering is not autoregressive.

For QM9, the model first samples the number of atoms $M$ from the empirical QM9 atom-count distribution, then samples $x,h$ conditional on $M$.

The training objective is the simplified diffusion noise-prediction loss. The paper notes that training uses $w(t)=1$ for stability, while exact log-likelihood evaluation uses the proper weighting.

## 2. Paper training time and cost

Appendix C gives the key training details.

For QM9:

```text
EGNN hidden features: 256
EGNN layers: 9
training epochs: 1100
training iterations: about 1.7 million
batch size: 64
diffusion steps: T = 1000
hardware: one NVIDIA GeForce GTX 1080 Ti
training time: about 7 days
sampling speed: 1.7 s/sample on GTX 1080 Ti
```

For GEOM-DRUGS:

```text
EGNN hidden features: 256
EGNN layers: 4
training epochs: 13
training iterations: about 1.2 million
batch size: 64
hardware: NVIDIA RTX A6000 GPUs
training time: about 5.5 days
sampling speed: 10.3 s/sample
```

Important detail: the README command uses `--n_epochs 3000`, but Appendix C says QM9 used 1100 epochs. The official bundled QM9 checkpoint has `current_epoch=1001`, so the artifact is closer to the appendix value than to the README value.

## 3. Reported QM9 results

Main QM9 table:

```text
Model       NLL              Atom stable    Mol stable
EDM         -110.7 ± 1.5     98.7 ± 0.1     82.0 ± 0.4
Data        --               99.0           95.2
```

QM9 validity table with hydrogens:

```text
EDM valid             91.9 ± 0.5 %
EDM valid and unique  90.7 ± 0.6 %
```

## 4. Hardware and software environment

PyTorch sees the AMD GPUs through ROCm as CUDA devices:

```text
torch 2.8.0a0+gitf443035.rocm631
cuda available True
device count 4
0 AMD Radeon Graphics
1 AMD Radeon Graphics
2 AMD Radeon Graphics
3 AMD Radeon Graphics
```

Installed missing dependency:

```bash
python -m pip install imageio
```

Already available:

```text
rdkit 2026.03.3
torch_geometric 2.6.1
ase 3.26.0
```

One compatibility patch was needed for current NumPy:

```text
EDM/qm9/data/prepare/qm9.py
```

Changed:

```python
dtype=np.int
```

to:

```python
dtype=int
```

Reason: `np.int` is removed in current NumPy.

## 5. Dataset discovery and workaround

The official EDM QM9 downloader uses Figshare URLs like:

```text
https://springernature.figshare.com/ndownloader/files/3195389
```

Those URLs currently return an AWS WAF challenge and zero-byte content in this environment:

```text
HTTP 202
x-amzn-waf-action: challenge
Content-Length: 0
```

The repo initially created invalid zero-byte files:

```text
EDM/qm9/temp/qm9/dsgdb9nsd.xyz.tar.bz2 0 bytes
EDM/qm9/temp/qm9/uncharacterized.txt 0 bytes
```

A local QM9 file was found:

```text
/g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/data/QM9/qm9_eV.npz
```

It contains 130831 molecules, which matches QM9 after removing the 3054 uncharacterized molecules. It stores:

```text
R, N, Z, id, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv, meta
```

This was converted into EDM’s expected split files:

```text
EDM/qm9/temp/qm9/train.npz  100000 molecules
EDM/qm9/temp/qm9/valid.npz   17748 molecules
EDM/qm9/temp/qm9/test.npz    13083 molecules
```

Total size:

```text
33M EDM/qm9/temp/qm9
```

The split uses the EDM/Cormorant split sizes and `np.random.seed(0)` permutation. Because the local NPZ already excludes the uncharacterized molecules, this should be the same molecule set as the official path, but it is not guaranteed to be byte-identical to the official downloader output.

The unconditional EDM model does not use the energy properties during training, so the missing thermochemical correction fields are not important for the unconditional reproduction. The repo prints warnings about missing thermochemical targets, but training and unconditional evaluation still run.

## 6. Smoke runs performed

### 6.1 Tiny CUDA smoke run

Command:

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

PYTHONUNBUFFERED=1 WANDB_MODE=disabled python main_qm9.py \
  --exp_name smoke_data_setup \
  --n_epochs 1 \
  --batch_size 8 \
  --nf 32 \
  --n_layers 2 \
  --diffusion_steps 10 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --normalize_factors [1,4,10] \
  --ema_decay 0 \
  --break_train_epoch True \
  --no_wandb \
  --save_model False \
  --dp False \
  --num_workers 0
```

Outcome:

- Dataset load succeeded.
- CUDA training step succeeded.
- The script then ran full validation and test because `main_qm9.py` evaluates at epoch 0 even after `break_train_epoch=True`.

### 6.2 Official-settings one-epoch smoke run

Command:

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

PYTHONUNBUFFERED=1 WANDB_MODE=disabled python main_qm9.py \
  --n_epochs 1 \
  --exp_name edm_qm9_official_1epoch_smoke \
  --n_stability_samples 100 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 64 \
  --nf 256 \
  --n_layers 9 \
  --lr 1e-4 \
  --normalize_factors [1,4,10] \
  --test_epochs 1 \
  --ema_decay 0.9999 \
  --no_wandb \
  --save_model True \
  --num_workers 0 \
  --n_report_steps 200
```

Outcome:

```text
Training using 4 GPUs
Epoch: 0, iter: 0/1563, Loss 2.84, NLL 2.84
Epoch: 0, iter: 200/1563, Loss 2.45, NLL 2.45
Epoch: 0, iter: 400/1563, Loss 2.57, NLL 2.57
Epoch: 0, iter: 600/1563, Loss 2.41, NLL 2.41
Epoch: 0, iter: 800/1563, Loss 2.70, NLL 2.70
Epoch: 0, iter: 1000/1563, Loss 2.57, NLL 2.57
Epoch: 0, iter: 1200/1563, Loss 2.57, NLL 2.57
Epoch: 0, iter: 1400/1563, Loss 2.50, NLL 2.50
Epoch took 226.7 seconds.
```

ROCm showed all 4 GPUs active:

```text
GPU use: 66, 56, 58, 58 %
VRAM allocated: about 3 % each
```

The job was stopped after the epoch during stability analysis. Reason: default evaluation with RDKit novelty first converts all QM9 train molecules to SMILES, which is slow and not needed to verify training throughput.

Projected training time from this run:

```text
226.7 s / epoch
1001 epochs ~= 63 h
1100 epochs ~= 69 h
3000 epochs ~= 189 h
```

This projection excludes evaluation overhead. With `test_epochs=20`, overhead is moderate but nonzero.

## 7. Pretrained checkpoint evaluation

The official repo includes a pretrained QM9 checkpoint:

```text
EDM/outputs/edm_qm9/args.pickle
EDM/outputs/edm_qm9/generative_model_ema.npy
```

Checkpoint args:

```text
nf=256
n_layers=9
diffusion_steps=1000
batch_size=64
normalize_factors=[1,4,10]
current_epoch=1001
```

### 7.1 NLL evaluation

Evaluated the pretrained checkpoint on the converted QM9 split.

Result:

```text
Val NLL:       -106.8574
Test NLL:      -111.4281  one pass
```

Paper:

```text
NLL: -110.7 ± 1.5
```

The test NLL reproduces the paper value.

### 7.2 Fresh sample stability

Sampled 1000 molecules from the pretrained checkpoint and evaluated stability without RDKit novelty conversion.

Result:

```text
n_samples:     1000
mol_stable:    0.8200
atom_stable:   0.9847
seconds:       280.6
```

Paper:

```text
mol_stable:    0.820 ± 0.004
atom_stable:   0.987 ± 0.001
```

Molecule stability matches exactly. Atom stability is slightly lower, likely due to version differences in bond thresholds, numeric behavior, or preprocessing.

### 7.3 Bundled official generated samples

The repo includes 10000 generated samples:

```text
EDM/generated_samples/samples_edm.zip
```

Evaluating all 10000 samples gave:

```text
n:                  10000
mol_stable:         0.818
atom_stable:        0.983678
valid:              0.9169
valid_unique:       0.9032
unique_given_valid: 0.9851
```

Paper:

```text
mol_stable:         0.820
atom_stable:        0.987
valid:              0.919
valid and unique:   0.907
```

This is a close reproduction. The persistent atom-stability difference is small and is probably not a model issue because molecule stability, validity, and NLL match.

## 8. Runbook

### 8.1 Check CUDA and dependencies

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('device count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

python - <<'PY'
for m in ['numpy', 'scipy', 'torch', 'torchvision', 'tqdm', 'wandb', 'imageio', 'rdkit']:
    try:
        mod = __import__(m)
        print(m, getattr(mod, '__version__', 'ok'))
    except Exception as e:
        print(m, 'missing', repr(e))
PY
```

### 8.2 Verify dataset files

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

find qm9/temp/qm9 -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

Expected:

```text
test.npz
train.npz
valid.npz
```

Ignore the zero-byte Figshare download artifacts if the NPZ split files exist.

### 8.3 Faithful paper-style training from scratch

Use this for the closest paper reproduction:

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

PYTHONUNBUFFERED=1 WANDB_MODE=disabled python main_qm9.py \
  --n_epochs 1100 \
  --exp_name edm_qm9_repro_1100ep \
  --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 64 \
  --nf 256 \
  --n_layers 9 \
  --lr 1e-4 \
  --normalize_factors [1,4,10] \
  --test_epochs 20 \
  --ema_decay 0.9999 \
  --no_wandb \
  --num_workers 0
```

### 8.4 Large-batch exploratory training

Because this node has much more GPU memory than the paper’s GTX 1080 Ti, a larger global batch is feasible. Important caveat: `main_qm9.py` uses one optimizer step per batch, so increasing `--batch_size` reduces the number of optimizer updates per epoch.

Paper update count:

```text
100000 training molecules / 64 batch ~= 1563 updates/epoch
1563 * 1100 ~= 1.72 million updates
```

With batch 256:

```text
100000 / 256 ~= 391 updates/epoch
1100 epochs ~= 0.43 million updates
```

So batch 256, LR 4e-4, 1100 epochs is a faster large-batch experiment, not a faithful optimizer-step reproduction. To match update count with batch 256, use about 4400 epochs.

Recommended large-batch first run:

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

PYTHONUNBUFFERED=1 WANDB_MODE=disabled python main_qm9.py \
  --n_epochs 1100 \
  --exp_name edm_qm9_b256_lr4e-4 \
  --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 256 \
  --nf 256 \
  --n_layers 9 \
  --lr 4e-4 \
  --normalize_factors [1,4,10] \
  --test_epochs 20 \
  --ema_decay 0.9999 \
  --no_wandb \
  --num_workers 0
```

If the loss becomes unstable, reduce LR to `2e-4`. If it is stable and underfits due to fewer updates, either increase epochs or go back to batch 64 for the faithful run.

### 8.5 Evaluate a trained model

Full official evaluation:

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

python eval_analyze.py \
  --model_path outputs/edm_qm9_repro_1100ep \
  --n_samples 10000 \
  --batch_size_gen 100 \
  --save_to_xyz True
```

This can be slow because RDKit novelty conversion builds SMILES for the full QM9 training set if the cache is missing.

Faster check:

- compute NLL on validation/test,
- compute atom/molecule stability on generated samples,
- skip novelty unless needed.

### 8.6 Standalone checkpoint metrics script

A local helper script was added:

```text
eval_checkpoint_metrics.py
```

It evaluates an existing checkpoint without touching a running training job. It reports:

- molecule-stable fraction,
- atom-stable fraction,
- RDKit validity,
- uniqueness among valid molecules,
- novelty among unique valid molecules,
- valid and unique fraction,
- optional saved XYZ-like molecule files.

Evaluate the current best checkpoint:

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

python eval_checkpoint_metrics.py \
  --model_path outputs/edm_qm9_b256_lr4e-4_1100ep \
  --n_samples 1000 \
  --batch_size_gen 100 \
  --csv_path outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/metrics.csv \
  --json_path outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/current.json
```

Evaluate a numbered epoch checkpoint:

```bash
python eval_checkpoint_metrics.py \
  --model_path outputs/edm_qm9_b256_lr4e-4_1100ep \
  --checkpoint_epoch 500 \
  --n_samples 1000 \
  --batch_size_gen 100 \
  --csv_path outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/metrics.csv \
  --json_path outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/epoch_500.json
```

Save sampled structures during metric evaluation. These are ASE-compatible `.extxyz` files:

```bash
python eval_checkpoint_metrics.py \
  --model_path outputs/edm_qm9_b256_lr4e-4_1100ep \
  --checkpoint_epoch 500 \
  --n_samples 1000 \
  --batch_size_gen 100 \
  --save_xyz \
  --save_dir outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/epoch_500_xyz \
  --csv_path outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/metrics.csv \
  --json_path outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/epoch_500.json
```

Fast stability-only check, no RDKit validity or novelty:

```bash
python eval_checkpoint_metrics.py \
  --model_path outputs/edm_qm9_b256_lr4e-4_1100ep \
  --checkpoint_epoch 500 \
  --n_samples 1000 \
  --batch_size_gen 100 \
  --no_rdkit
```

This script is useful because the upstream training loop logs `mol_stable` and `atom_stable` only to wandb. With `--no_wandb`, those two metrics are not printed in the training log.

Checkpoint sweep over 100-epoch intervals. The loop skips missing checkpoints. At the time of this sweep, the available 100-multiple checkpoints were 100, 200, 400, and 500. Epoch 300 was not present because checkpoint files are only saved on evaluated epochs when validation NLL improves.

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

OUT=outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/checkpoint_metrics_100_200_400_500.csv
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

for E in 100 200 300 400 500 680 760 880 940 1000; do
  CKPT=outputs/edm_qm9_b256_lr4e-4_1100ep/generative_model_ema_${E}.npy
  if [ ! -f "$CKPT" ]; then
    echo "missing checkpoint ${E}, skipping"
    continue
  fi

  CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python eval_checkpoint_metrics.py \
    --model_path outputs/edm_qm9_b256_lr4e-4_1100ep \
    --checkpoint_epoch "$E" \
    --n_samples 1000 \
    --batch_size_gen 1000 \
    --csv_path "$OUT" \
    --json_path outputs/edm_qm9_b256_lr4e-4_1100ep/eval_metrics/epoch_${E}_1000.json
done
```

Results from 1000 generated samples per checkpoint:

| epoch | mol stable | atom stable | valid | unique given valid | valid and unique | novelty given unique | unique valid count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.002 | 0.738 | 0.203 | 1.000 | 0.203 | 1.000 | 203 |
| 200 | 0.058 | 0.849 | 0.534 | 1.000 | 0.534 | 0.987 | 534 |
| 400 | 0.150 | 0.893 | 0.553 | 1.000 | 0.553 | 0.948 | 553 |
| 500 | 0.194 | 0.902 | 0.594 | 0.998 | 0.593 | 0.943 | 593 |

These metrics are stochastic because they are computed from newly sampled molecules. The trends are still useful: atom stability, molecule stability, and validity improve with training, but the large-batch run at epoch 500 is still far from the paper target of about 0.82 molecule stability, 0.987 atom stability, and 0.907 valid-and-unique.

### 8.7 Parse training log metrics

The training log does print RDKit validity, uniqueness, novelty, validation NLL, and test NLL every `TEST_EPOCHS` epochs.

Use this parser:

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

export LOG=logs/edm_qm9_b256_lr4e-4_1100ep_20260719_213636.log

python - <<'PY'
import os, re
log = os.environ.get('LOG')
rows = []
cur = {}
for line in open(log, errors='replace'):
    m = re.search(r'Analyzing molecule stability at epoch (\d+)', line)
    if m:
        if cur:
            rows.append(cur)
        cur = {'epoch': int(m.group(1))}
    m = re.search(r'Validity over (\d+) molecules: ([0-9.]+)%', line)
    if m and cur:
        cur['n'] = int(m.group(1))
        cur['valid_pct'] = float(m.group(2))
    m = re.search(r'Uniqueness over (\d+) valid molecules: ([0-9.]+)%', line)
    if m and cur:
        cur['unique_pct_given_valid'] = float(m.group(2))
    m = re.search(r'Novelty over (\d+) unique valid molecules: ([0-9.]+)%', line)
    if m and cur:
        cur['novel_pct_given_unique'] = float(m.group(2))
    m = re.search(r'Val loss: *([-0-9.]+)\s+Test loss: *([-0-9.]+)', line)
    if m and cur:
        cur['val_nll'] = float(m.group(1))
        cur['test_nll'] = float(m.group(2))
if cur:
    rows.append(cur)

print('epoch,n,valid_pct,unique_pct_given_valid,valid_unique_pct,novel_pct_given_unique,val_nll,test_nll')
for r in rows:
    valid_unique = r.get('valid_pct', 0) * r.get('unique_pct_given_valid', 0) / 100
    print(f"{r.get('epoch')},{r.get('n','')},{r.get('valid_pct','')},{r.get('unique_pct_given_valid','')},{valid_unique:.2f},{r.get('novel_pct_given_unique','')},{r.get('val_nll','')},{r.get('test_nll','')}")
PY
```

As of the active large-batch run at epoch 500, the parsed metrics were:

```text
epoch valid% unique_given_valid% valid_unique% novelty_given_unique% val_nll  test_nll
400   55.7   100.0               55.70         95.87                 -44.87   -40.71
420   54.1    99.82              54.00         95.37                 -39.31   -36.76
440   56.5   100.0               56.50         95.40                 -49.56   -40.04
460   60.3    99.83              60.20         95.35                 -46.01   -47.70
480   60.0   100.0               60.00         95.00                 -38.33   -48.32
500   59.7    99.83              59.60         94.97                 -55.85   -48.54
```

The paper target is roughly:

```text
valid:             91.9 %
valid and unique:  90.7 %
mol stable:        82.0 %
atom stable:       98.7 %
NLL:              -110.7
```

## 9. Queue script

A queue script was written to:

```text
EDM/submit_edm_qm9_repro.sh
```

Submit with Flux:

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM
flux batch -q pbatch -t 72h -N 1 -n 1 -g 4 ./submit_edm_qm9_repro.sh
```

Default settings in the script:

```text
BATCH_SIZE=256
LR=4e-4
EPOCHS=1100
TEST_EPOCHS=20
N_STABILITY_SAMPLES=1000
```

Override example:

```bash
BATCH_SIZE=512 LR=8e-4 EPOCHS=1100 \
flux batch -q pbatch -t 72h -N 1 -n 1 -g 4 ./submit_edm_qm9_repro.sh
```

For a faithful paper-style run:

```bash
BATCH_SIZE=64 LR=1e-4 EPOCHS=1100 \
flux batch -q pbatch -t 72h -N 1 -n 1 -g 4 ./submit_edm_qm9_repro.sh
```

For step-matched large-batch training:

```bash
BATCH_SIZE=256 LR=4e-4 EPOCHS=4400 \
flux batch -q pbatch -t 240h -N 1 -n 1 -g 4 ./submit_edm_qm9_repro.sh
```

## 10. Notable caveats and fixes

1. Figshare download is blocked by WAF in this environment. The local QM9 NPZ workaround was used.
2. The converted dataset has no thermochemical `_thermo` fields. This matters for some property workflows, but not for unconditional molecule generation.
3. `main_qm9.py` uses `torch.nn.DataParallel`, not DDP. One Python process controls all visible GPUs.
4. Larger batch size changes the number of optimizer updates per epoch. Do not compare batch 256 for 1100 epochs directly to the paper’s batch 64 for 1100 epochs.
5. RDKit novelty evaluation is slow if the QM9 SMILES cache is absent. After the first conversion, the cache is:

```text
qm9/temp/qm9_smiles.pickle
```

In this run it was created and is about 5.6 MB.

6. The official README says 3000 epochs, while Appendix C says 1100 epochs and the bundled checkpoint is at epoch 1001.
7. The atom-stability number in this environment is slightly below the paper, but NLL, molecule stability, validity, and valid-unique metrics reproduce closely.
8. Flux copies the job script to `/var/tmp` and starts there on this system. The queue script therefore must not use `${BASH_SOURCE[0]}` and must not assume the current directory is the repo. `submit_edm_qm9_repro.sh` sets:

```bash
PROJECT_ROOT=${PROJECT_ROOT:-/g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM}
cd "$PROJECT_ROOT"
```

9. The upstream code wrote nonstandard XYZ-like `.txt` files and rendered PNG/GIF graphics. This fork now writes ASE-compatible `.extxyz` files through `ase.io` and disables PNG/GIF rendering. Molecule samples are written as separate `.extxyz` files. Chains and conditional sweeps are written as multi-frame `.extxyz` trajectories.
10. Training-loop stability evaluation does not save the 1000 evaluation samples by default. It only uses them for metrics. Visualization samples are saved every `TEST_EPOCHS` epochs under directories like:

```text
outputs/<exp_name>/epoch_500_/molecule_000.extxyz
outputs/<exp_name>/epoch_500_/chain/chain.extxyz
```

11. The best checkpoint is overwritten at:

```text
outputs/<exp_name>/generative_model.npy
outputs/<exp_name>/generative_model_ema.npy
outputs/<exp_name>/optim.npy
outputs/<exp_name>/args.pickle
```

Numbered checkpoints are saved only on evaluated epochs when validation NLL improves:

```text
outputs/<exp_name>/generative_model_500.npy
outputs/<exp_name>/generative_model_ema_500.npy
outputs/<exp_name>/optim_500.npy
outputs/<exp_name>/args_500.pickle
```
