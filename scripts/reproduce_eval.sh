#!/usr/bin/env bash
set -euo pipefail

# Evaluation wrapper. It consumes checkpoints/{CLIP,CDM,VLM} and regenerates
# the small JSON result files used by the figure notebooks.

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

EVAL_SCRIPTS=(
  "figures/eval-clip-risk.py"
  "figures/eval-zsc-risk.py"
  "figures/eval-cdm-risk.py"
  "figures/eval-vlm-risk.py"
  "figures/eval-clip-ood.py"
  "figures/eval-zsc-ood.py"
  "figures/eval-cdm-ood.py"
  "figures/eval-vlm-ood.py"
  "figures/eval-zsc-numsamples.py"
)

EXPECTED_JSON=(
  "figures/data/ghm-data/clip-risk.json"
  "figures/data/ghm-data/zsc-risk.json"
  "figures/data/ghm-data/cdm-risk.json"
  "figures/data/ghm-data/vlm-risk.json"
  "figures/data/ghm-data/clip-ood.json"
  "figures/data/ghm-data/zsc-ood.json"
  "figures/data/ghm-data/cdm-ood.json"
  "figures/data/ghm-data/cdm-ood-pt20.json"
  "figures/data/ghm-data/vlm-ood.json"
  "figures/data/ghm-data/vlm-ood-pi20.json"
  "figures/data/ghm-data/zsc-numsamples.json"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: skipping checkpoint existence checks."
else
  for family in CLIP CDM VLM; do
    if [[ ! -d "checkpoints/${family}" ]]; then
      echo "ERROR: checkpoints/${family} is missing."
      echo "Run python scripts/download_ckpt.py or scripts/reproduce_train.sh first."
      exit 1
    fi
  done
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p figures/data/ghm-data
fi

for script_path in "${EVAL_SCRIPTS[@]}"; do
  echo
  echo "==> Evaluation stage: ${script_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: would run ${PYTHON} ${script_path}"
  else
    "${PYTHON}" "${script_path}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN=1: would check generated JSON files:"
  printf '  %s\n' "${EXPECTED_JSON[@]}"
  exit 0
fi

echo
echo "==> Checking generated JSON files"
for json_path in "${EXPECTED_JSON[@]}"; do
  if [[ ! -s "${json_path}" ]]; then
    echo "ERROR: expected JSON was not generated: ${json_path}"
    exit 1
  fi
  echo "ok: ${json_path}"
done

echo
echo "Evaluation reproduction complete."
