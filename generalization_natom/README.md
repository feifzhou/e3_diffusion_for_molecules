# EDM fixed-size generalization scan

This scan tests the official pretrained QM9 EDM checkpoint under fixed total atom counts outside the training support.

Checkpoint:

```text
outputs/edm_qm9/generative_model_ema.npy
```

QM9 training support:

```text
maximum total atoms: 29
maximum heavy atoms: 9
```

The scan fixes total atom count `n_atoms` from 21 to 45. For each total size, it generates 1000 molecules in one batch and computes:

- `mol_stable`: fraction of generated molecules where every atom has an allowed valence under the repo's distance-threshold bond perception.
- `atom_stable`: fraction of atoms with allowed valence under the same bond perception.
- `valid`: RDKit validity after inferring bonds from distances and running `Chem.SanitizeMol`.
- `unique valid`: `validity * uniqueness_given_valid`, where uniqueness is computed over valid SMILES.
- `mean heavy`: mean number of non-H atoms in the generated samples.

Important caveat: RDKit validity is not the same as molecular stability. RDKit validity is computed after distance-based bond inference and fragment handling. In particular, all-H collapse can appear RDKit-valid but chemically useless, with zero molecule stability and near-zero uniqueness.

## Run command

```bash
cd /g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM

rm -rf generalization_natom
mkdir -p generalization_natom

CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python eval_size_generalization.py \
  --model_path outputs/edm_qm9 \
  --n_min 21 \
  --n_max 45 \
  --n_samples 1000 \
  --batch_size 1000 \
  --save_dir generalization_natom \
  --csv_path generalization_natom/size_generalization_21_45_n1000.csv \
  --json_path generalization_natom/size_generalization_21_45_n1000.json \
  > generalization_natom/run.log 2>&1
```

Outputs:

```text
generalization_natom/README.md
generalization_natom/run.log
generalization_natom/size_generalization_21_45_n1000.csv
generalization_natom/size_generalization_21_45_n1000.json
generalization_natom/n_atoms_21.extxyz
generalization_natom/n_atoms_22.extxyz
...
generalization_natom/n_atoms_45.extxyz
```

Each `n_atoms_XX.extxyz` is a multi-frame extxyz with 1000 generated molecules of fixed total atom count `XX`.

## Results

| n_atoms | mean heavy | mol stable | atom stable | valid | unique valid | H frac | C frac | N frac | O frac | F frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | 8.99 | 0.882 | 0.992 | 0.958 | 0.940 | 0.572 | 0.336 | 0.035 | 0.057 | 0.000 |
| 22 | 8.99 | 0.887 | 0.993 | 0.962 | 0.946 | 0.591 | 0.325 | 0.038 | 0.046 | 0.000 |
| 23 | 9.09 | 0.900 | 0.994 | 0.965 | 0.930 | 0.605 | 0.323 | 0.027 | 0.045 | 0.000 |
| 24 | 9.11 | 0.926 | 0.996 | 0.971 | 0.917 | 0.620 | 0.312 | 0.032 | 0.036 | 0.000 |
| 25 | 9.20 | 0.915 | 0.995 | 0.974 | 0.875 | 0.632 | 0.313 | 0.021 | 0.034 | 0.000 |
| 26 | 9.29 | 0.904 | 0.994 | 0.977 | 0.866 | 0.643 | 0.303 | 0.027 | 0.027 | 0.000 |
| 27 | 9.35 | 0.897 | 0.993 | 0.980 | 0.726 | 0.654 | 0.302 | 0.021 | 0.023 | 0.000 |
| 28 | 9.60 | 0.875 | 0.992 | 0.971 | 0.762 | 0.657 | 0.299 | 0.020 | 0.024 | 0.000 |
| 29 | 9.66 | 0.822 | 0.987 | 0.969 | 0.641 | 0.667 | 0.296 | 0.020 | 0.016 | 0.000 |
| 30 | 9.94 | 0.762 | 0.981 | 0.967 | 0.669 | 0.669 | 0.298 | 0.013 | 0.020 | 0.000 |
| 31 | 10.01 | 0.557 | 0.964 | 0.955 | 0.741 | 0.677 | 0.293 | 0.019 | 0.011 | 0.000 |
| 32 | 10.03 | 0.465 | 0.948 | 0.942 | 0.625 | 0.686 | 0.296 | 0.013 | 0.004 | 0.000 |
| 33 | 10.15 | 0.147 | 0.923 | 0.904 | 0.711 | 0.693 | 0.296 | 0.008 | 0.003 | 0.000 |
| 34 | 10.24 | 0.167 | 0.906 | 0.899 | 0.786 | 0.699 | 0.295 | 0.006 | 0.001 | 0.000 |
| 35 | 10.36 | 0.158 | 0.893 | 0.841 | 0.703 | 0.704 | 0.292 | 0.004 | 0.000 | 0.000 |
| 36 | 10.33 | 0.012 | 0.857 | 0.813 | 0.676 | 0.713 | 0.284 | 0.003 | 0.000 | 0.000 |
| 37 | 10.14 | 0.025 | 0.816 | 0.732 | 0.556 | 0.726 | 0.271 | 0.003 | 0.000 | 0.000 |
| 38 | 0.00 | 0.000 | 0.000 | 1.000 | 0.002 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 39 | 8.58 | 0.000 | 0.645 | 0.653 | 0.275 | 0.780 | 0.218 | 0.002 | 0.000 | 0.000 |
| 40 | 0.00 | 0.000 | 0.001 | 1.000 | 0.002 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 41 | 5.95 | 0.000 | 0.434 | 0.546 | 0.120 | 0.855 | 0.140 | 0.001 | 0.004 | 0.000 |
| 42 | 0.00 | 0.000 | 0.000 | 1.000 | 0.002 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 43 | 5.31 | 0.000 | 0.295 | 0.447 | 0.078 | 0.877 | 0.082 | 0.002 | 0.040 | 0.000 |
| 44 | 4.98 | 0.000 | 0.254 | 0.396 | 0.053 | 0.887 | 0.061 | 0.002 | 0.050 | 0.000 |
| 45 | 2.39 | 0.000 | 0.192 | 0.443 | 0.011 | 0.947 | 0.005 | 0.003 | 0.045 | 0.000 |

## Interpretation

- The model works well in distribution, `n_atoms = 21` to `29`.
- It extrapolates weakly to `n_atoms = 30`, with `mol_stable = 0.762` and `atom_stable = 0.981`.
- Stability degrades sharply after 30 atoms.
- By 33 to 35 atoms, RDKit validity remains moderate but molecule stability is poor. The model is producing structures that can often be sanitized after bond perception, but most atoms do not satisfy the strict valence-count stability check.
- At 38, 40, and 42 atoms the model collapses to all-H samples. RDKit validity is 1.0 there, but this is meaningless. `mol_stable = 0`, `atom_stable ≈ 0`, and `unique valid ≈ 0`.
- The heavy atom count does not scale to 15. It saturates near 10 heavy atoms, then collapses for larger total atom counts.
- Sampling emitted many `Warning: detected nan, resetting EGNN output to zero` messages for large out-of-distribution sizes. This is a numerical failure mode of the pretrained denoiser outside the QM9 size range.
