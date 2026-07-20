#!/usr/bin/env bash
# Flux submission script for EDM QM9 training on one 4-GPU node.
# Submit from EDM/ with:
#   flux batch -q pbatch -t 72h -N 1 -n 1 -g 4 ./submit_edm_qm9_repro.sh
#
# Defaults are intentionally large-batch exploratory settings, not the exact paper run.
# Override at submit time, e.g.:
#   BATCH_SIZE=64 LR=1e-4 EPOCHS=1100 flux batch -q pbatch -t 72h -N 1 -n 1 -g 4 ./submit_edm_qm9_repro.sh

#flux: -N 1
#flux: -t 72h
#flux: -q pbatch

set -euo pipefail

# Flux runs this batch script from a per-job /var/tmp directory on this system.
# The copied script path and initial working directory are irrelevant.
# Always enter the EDM repository explicitly.
PROJECT_ROOT=${PROJECT_ROOT:-/g/g90/zhou6/lassen-space/NPS/notebooks/GCparticle/EDM}
cd "$PROJECT_ROOT"

# Environment.
export PATH=/usr/WS2/zhou6/tuopt/bin:$PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}

# Make all four GPUs visible. main_qm9.py uses torch.nn.DataParallel when --dp True.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}

# Hyperparameters.
# Paper-faithful values are BATCH_SIZE=64, LR=1e-4, EPOCHS=1100.
# These defaults use linear LR scaling for 4x larger global batch.
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-4e-4}
EPOCHS=${EPOCHS:-1100}
TEST_EPOCHS=${TEST_EPOCHS:-20}
N_STABILITY_SAMPLES=${N_STABILITY_SAMPLES:-1000}
NUM_WORKERS=${NUM_WORKERS:-0}
NF=${NF:-256}
N_LAYERS=${N_LAYERS:-9}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-1000}
EMA_DECAY=${EMA_DECAY:-0.9999}
EXP_NAME=${EXP_NAME:-edm_qm9_b${BATCH_SIZE}_lr${LR}_${EPOCHS}ep}

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

cat <<EOF
==========================================
EDM QM9 training
==========================================
Date:                 $(date)
Host:                 $(hostname)
Working directory:    $(pwd)
Python:               $(which python)
Experiment:           $EXP_NAME
Batch size, global:   $BATCH_SIZE
LR:                   $LR
Epochs:               $EPOCHS
Test epochs:          $TEST_EPOCHS
Stability samples:    $N_STABILITY_SAMPLES
Visible GPUs:         CUDA=$CUDA_VISIBLE_DEVICES HIP=$HIP_VISIBLE_DEVICES
Log file:             $LOG_FILE
==========================================
EOF

python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('device count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

if [ ! -f qm9/temp/qm9/train.npz ] || [ ! -f qm9/temp/qm9/valid.npz ] || [ ! -f qm9/temp/qm9/test.npz ]; then
    echo "ERROR: EDM QM9 split files are missing under qm9/temp/qm9." >&2
    echo "Expected train.npz, valid.npz, test.npz. See EDM_reproduction_report.md for the conversion workaround." >&2
    exit 1
fi

set -x
python main_qm9.py \
    --n_epochs "$EPOCHS" \
    --exp_name "$EXP_NAME" \
    --n_stability_samples "$N_STABILITY_SAMPLES" \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --diffusion_loss_type l2 \
    --batch_size "$BATCH_SIZE" \
    --nf "$NF" \
    --n_layers "$N_LAYERS" \
    --lr "$LR" \
    --normalize_factors "[1,4,10]" \
    --test_epochs "$TEST_EPOCHS" \
    --ema_decay "$EMA_DECAY" \
    --no_wandb \
    --num_workers "$NUM_WORKERS" \
    --n_report_steps 100 \
    2>&1 | tee "$LOG_FILE"
set +x

cat <<EOF
==========================================
EDM QM9 training finished
Date:       $(date)
Experiment: outputs/$EXP_NAME
Log file:   $LOG_FILE
==========================================
EOF
