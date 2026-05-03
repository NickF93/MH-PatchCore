#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${ROOT_DIR}/run_mhpc.py"

FAIL_FAST=0
DRY_RUN=0
CONTINUE_EXISTING=0
ONLY_BASELINES=0
ONLY_ABLATIONS=0
INCLUDE_GLOBS=()
EXCLUDE_GLOBS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all_experiments.sh [options]

Runs MVTec configs in deterministic order:
1) configs/mvtec/baselines/**/*.yaml
2) configs/mvtec/ablations/**/*.yaml

Options:
  --all                      Run baselines and ablations (default).
  --baselines                Run only baseline configs.
  --ablations                Run only ablation configs.
  --continue                 Skip configs with at least one completed run.
  --fail-fast                Stop at the first failed config.
  --dry-run                  Print selected configs and exit.
  --include-glob PATTERN...  Include only configs matching shell-style globs.
  --exclude-glob PATTERN...  Exclude configs matching shell-style globs.
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      ONLY_BASELINES=0
      ONLY_ABLATIONS=0
      shift
      ;;
    --baselines)
      ONLY_BASELINES=1
      ONLY_ABLATIONS=0
      shift
      ;;
    --ablations)
      ONLY_ABLATIONS=1
      ONLY_BASELINES=0
      shift
      ;;
    --continue)
      CONTINUE_EXISTING=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --include-glob)
      shift
      if [[ $# -eq 0 || "$1" == -* ]]; then
        echo "Option requires at least one pattern: --include-glob" >&2
        usage
        exit 2
      fi
      while [[ $# -gt 0 && "$1" != -* ]]; do
        INCLUDE_GLOBS+=("$1")
        shift
      done
      ;;
    --exclude-glob)
      shift
      if [[ $# -eq 0 || "$1" == -* ]]; then
        echo "Option requires at least one pattern: --exclude-glob" >&2
        usage
        exit 2
      fi
      while [[ $# -gt 0 && "$1" != -* ]]; do
        EXCLUDE_GLOBS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}" >&2
  exit 2
fi

matches_any_glob() {
  local rel_cfg="$1"
  shift

  local pattern
  for pattern in "$@"; do
    if [[ "${rel_cfg}" == ${pattern} ]]; then
      return 0
    fi
  done
  return 1
}

matches_optional_filters() {
  local rel_cfg="$1"
  if [[ ${#INCLUDE_GLOBS[@]} -gt 0 ]] && ! matches_any_glob "${rel_cfg}" "${INCLUDE_GLOBS[@]}"; then
    return 1
  fi
  if [[ ${#EXCLUDE_GLOBS[@]} -gt 0 ]] && matches_any_glob "${rel_cfg}" "${EXCLUDE_GLOBS[@]}"; then
    return 1
  fi
  return 0
}

read_config_run_base() {
  local cfg_path="$1"
  "${PYTHON_BIN}" - "${ROOT_DIR}" "${cfg_path}" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
cfg_path = Path(sys.argv[2]).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from mhpc.eval.config import load_run_config

cfg = load_run_config(cfg_path)
print(cfg.paths.output_root)
print(cfg.experiment.name)
PY
}

has_completed_run() {
  local cfg_path="$1"
  local preflight_out output_root experiment_name experiment_root
  if ! preflight_out="$(read_config_run_base "${cfg_path}")"; then
    return 1
  fi
  output_root="$(printf '%s\n' "${preflight_out}" | sed -n '1p')"
  experiment_name="$(printf '%s\n' "${preflight_out}" | sed -n '2p')"
  experiment_root="${output_root}/${experiment_name}"
  [[ -n "$(find "${experiment_root}" -mindepth 2 -maxdepth 2 -path '*/metrics/summary.csv' -print -quit 2>/dev/null)" ]]
}

CONFIG_SEARCH_ROOTS=()
if [[ "${ONLY_ABLATIONS}" -eq 0 ]]; then
  CONFIG_SEARCH_ROOTS+=("${ROOT_DIR}/configs/mvtec/baselines")
fi
if [[ "${ONLY_BASELINES}" -eq 0 ]]; then
  CONFIG_SEARCH_ROOTS+=("${ROOT_DIR}/configs/mvtec/ablations")
fi

mapfile -t CONFIG_CANDIDATES < <(
  for search_root in "${CONFIG_SEARCH_ROOTS[@]}"; do
    LC_ALL=C find "${search_root}" -type f -name '*.yaml' | LC_ALL=C sort
  done
)

ORDERED_CONFIGS=()
for cfg in "${CONFIG_CANDIDATES[@]}"; do
  rel_cfg="${cfg#${ROOT_DIR}/}"
  if ! matches_optional_filters "${rel_cfg}"; then
    continue
  fi
  if [[ "${CONTINUE_EXISTING}" -eq 1 ]] && has_completed_run "${cfg}"; then
    continue
  fi
  ORDERED_CONFIGS+=("${cfg}")
done

if [[ ${#ORDERED_CONFIGS[@]} -eq 0 ]]; then
  echo "No runnable configs found." >&2
  exit 0
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%s\n' "${ORDERED_CONFIGS[@]#${ROOT_DIR}/}"
  exit 0
fi

BATCH_TS="$(date -u +%Y%m%d_%H%M%S)"
BATCH_LOG_ROOT="${ROOT_DIR}/logs/batch_runs/${BATCH_TS}"
mkdir -p "${BATCH_LOG_ROOT}"

echo "Batch log root: ${BATCH_LOG_ROOT}"
echo "Python executable: ${PYTHON_BIN}"
echo "Total configs: ${#ORDERED_CONFIGS[@]}"

FAILURES=0
INDEX=0

for cfg_abs in "${ORDERED_CONFIGS[@]}"; do
  INDEX=$((INDEX + 1))
  cfg_rel="${cfg_abs#${ROOT_DIR}/}"
  log_file="${BATCH_LOG_ROOT}/$(printf '%03d' "${INDEX}")__$(basename "${cfg_abs%.yaml}").log"

  echo
  echo "=== [${INDEX}/${#ORDERED_CONFIGS[@]}] Running ${cfg_rel} ==="

  if "${PYTHON_BIN}" "${RUNNER}" --config "${cfg_abs}" 2>&1 | tee "${log_file}"; then
    echo "OK: ${cfg_rel}"
  else
    FAILURES=$((FAILURES + 1))
    echo "FAILED: ${cfg_rel}"
    if [[ "${FAIL_FAST}" -eq 1 ]]; then
      break
    fi
  fi
done

if [[ "${FAILURES}" -gt 0 ]]; then
  exit 1
fi

exit 0
