#!/usr/bin/env bash
set -euo pipefail

# Figure rendering wrapper. It consumes JSON files in figures/data/ghm-data and
# the ImageNet tensor in figures/data/imagenet-data, then executes the plotting
# notebooks to produce PDFs in figures/output.

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
NOTEBOOK_TIMEOUT="${NOTEBOOK_TIMEOUT:-2400}"
JUPYTER_KERNEL="${JUPYTER_KERNEL:-python3}"

REQUIRED_INPUTS=(
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
  "figures/data/imagenet-data/similarities_targets.pt"
)

NOTEBOOKS=(
  "Fig2-taskRisks.ipynb"
  "Fig56-ZSC-m.ipynb"
  "Fig7-imagenet.ipynb"
  "Fig8-OODRisks.ipynb"
  "Fig9-OODrisks2.ipynb"
)

EXPECTED_PDFS=(
  "figures/output/Fig2-a-CLIP-risk.pdf"
  "figures/output/Fig2-b-ZSC-risk.pdf"
  "figures/output/Fig2-c-CDM-risk.pdf"
  "figures/output/Fig2-d-VLM-risk.pdf"
  "figures/output/Fig5-ZSC-vs-M.pdf"
  "figures/output/Fig6-a-ZSC-Fit-Standard TF.pdf"
  "figures/output/Fig6-b-ZSC-Fit-Guided TF.pdf"
  "figures/output/Fig6-c-ZSC-Fit-Shallow TF.pdf"
  "figures/output/Fig7-a-imagenet-loss.pdf"
  "figures/output/Fig7-b-imagenet-acc1.pdf"
  "figures/output/Fig7-c-imagenet-acc5.pdf"
  "figures/output/Fig8-a-CLIP-ood.pdf"
  "figures/output/Fig8-b-ZSC-ood.pdf"
  "figures/output/Fig8-c-CDM-ood.pdf"
  "figures/output/Fig8-d-VLM-ood.pdf"
  "figures/output/Fig9-a-CDM-ood.pdf"
  "figures/output/Fig9-b-VLM-ood.pdf"
)

echo "==> Checking figure inputs"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: would require:"
  printf '  %s\n' "${REQUIRED_INPUTS[@]}"
else
  for input_path in "${REQUIRED_INPUTS[@]}"; do
    if [[ ! -s "${input_path}" ]]; then
      echo "ERROR: missing required input: ${input_path}"
      echo "Run scripts/reproduce_eval.sh and python scripts/download_data.py first."
      exit 1
    fi
    echo "ok: ${input_path}"
  done
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN=1: would render notebooks:"
  printf '  figures/%s\n' "${NOTEBOOKS[@]}"
  echo
  echo "DRY_RUN=1: would check PDFs:"
  printf '  %s\n' "${EXPECTED_PDFS[@]}"
  exit 0
fi

mkdir -p figures/output

pushd figures >/dev/null
for notebook in "${NOTEBOOKS[@]}"; do
  echo
  echo "==> Rendering notebook: figures/${notebook}"
  "${PYTHON}" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace "${notebook}" \
    --ExecutePreprocessor.kernel_name="${JUPYTER_KERNEL}" \
    --ExecutePreprocessor.timeout="${NOTEBOOK_TIMEOUT}"
done
popd >/dev/null

echo
echo "==> Checking rendered PDF files"
for pdf_path in "${EXPECTED_PDFS[@]}"; do
  if [[ ! -s "${pdf_path}" ]]; then
    echo "ERROR: expected PDF was not generated: ${pdf_path}"
    exit 1
  fi
  echo "ok: ${pdf_path}"
done

echo
echo "Figure reproduction complete. Outputs are in figures/output/."
