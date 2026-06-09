#!/usr/bin/env bash
set -euo pipefail

# Full retraining wrapper. It runs the experiment scripts that create model
# artifacts for the paper figures, then stages logs/{CLIP,CDM,VLM} into
# checkpoints/{CLIP,CDM,VLM}, which is the location used by reproduction evals.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
  elif [[ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
    PYTHON="${REPO_ROOT}/.venv/Scripts/python.exe"
  else
    echo "ERROR: Python not found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi
DRY_RUN="${DRY_RUN:-0}"

TRAIN_SCRIPTS=(
  "scripts/experiments/exp_clip_standardTF.sh"
  "scripts/experiments/exp_clip_guidedTF.sh"
  "scripts/experiments/exp_clip_shallowTF.sh"
  "scripts/experiments/exp_cdm_standardTF.sh"
  "scripts/experiments/exp_cdm_guidedTF.sh"
  "scripts/experiments/exp_cdm_shallowTF.sh"
  "scripts/experiments/exp_cdm_jointtrain.sh"
  "scripts/experiments/exp_vlm_standardTF.sh"
  "scripts/experiments/exp_vlm_guidedTF.sh"
  "scripts/experiments/exp_vlm_shallowTF.sh"
  "scripts/experiments/exp_vlm_jointtrain.sh"
)

run_step() {
  local script_path="$1"
  echo
  echo "==> Training stage: ${script_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: would run bash ${script_path}"
  else
    bash "${script_path}"
  fi
}

stage_logs_to_checkpoints() {
  echo
  echo "==> Moving completed training artifacts into checkpoints/"
  mkdir -p checkpoints
  for family in CLIP CDM VLM; do
    if [[ ! -d "logs/${family}" ]]; then
      echo "WARNING: logs/${family} not found; skipping."
      continue
    fi
    mkdir -p "checkpoints/${family}"
    cp -a "logs/${family}/." "checkpoints/${family}/"
    rm -rf "logs/${family}"
    echo "moved logs/${family} -> checkpoints/${family}"
  done
}

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" - <<'PY'
import ghmclip
print(f"Using ghmclip from {ghmclip.__file__}")
PY
fi

echo "This script starts long-running training jobs. Use python scripts/download_ckpt.py to skip retraining."
for script_path in "${TRAIN_SCRIPTS[@]}"; do
  run_step "${script_path}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN=1: would move logs/{CLIP,CDM,VLM} into checkpoints/ after training."
else
  stage_logs_to_checkpoints
fi

echo
echo "Training reproduction complete. Checkpoints are under checkpoints/{CLIP,CDM,VLM}."
