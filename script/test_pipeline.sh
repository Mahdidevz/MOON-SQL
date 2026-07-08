#!/usr/bin/env bash
# =============================================================================
# script/test_pipeline.sh — Dynamic test wrapper for the MOON-SQL pipeline.
#
# Usage:
#   bash script/test_pipeline.sh -n 50   # run on first 50 samples
#   bash script/test_pipeline.sh 50      # shorthand — positional N also works
#   bash script/test_pipeline.sh --full  # run on the complete dev.json
#   bash script/test_pipeline.sh         # same as --full (safe default)
#
# The script calls src/utils/slice_data.py to prepare the dataset slice, then
# wires the resolved paths into run.sh (generation) and eval.sh (evaluation)
# via environment variables — without touching any core pipeline logic.
# =============================================================================
set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────────
# SCRIPT_DIR  = .../MOON-SQL/script/
# PROJECT_ROOT = .../MOON-SQL/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SLICER="$PROJECT_ROOT/src/utils/slice_data.py"
PYTHON="${PYTHON:-python3}"      # override with: PYTHON=python bash script/test_pipeline.sh

# ── Colours (disabled automatically when stdout is not a TTY) ─────────────────
if [[ -t 1 ]]; then
    BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'
    YELLOW='\033[1;33m'; RESET='\033[0m'
else
    BOLD=''; CYAN=''; GREEN=''; YELLOW=''; RESET=''
fi

banner() { echo -e "${BOLD}${CYAN}$*${RESET}"; }
info()   { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()   { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
err()    { echo -e "\033[0;31m[ERROR]\033[0m $*" >&2; }

# ── Argument parsing ───────────────────────────────────────────────────────────
NUM_SAMPLES=""
USE_FULL=false

usage() {
    cat <<EOF
Usage: bash script/test_pipeline.sh [OPTIONS]

Options:
  -n <N>     Use the first N samples from dev.json (integer > 0)
  --full     Use the complete dev.json (default when no option is given)
  -h, --help Show this help message

Positional shorthand:
  bash script/test_pipeline.sh 50    # equivalent to -n 50

Examples:
  bash script/test_pipeline.sh -n 50        # 50-sample smoke test
  bash script/test_pipeline.sh -n 200       # larger subset
  bash script/test_pipeline.sh --full       # full evaluation run
  bash script/test_pipeline.sh              # same as --full
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)
            if [[ -z "${2:-}" ]]; then
                err "-n requires a positive integer argument."
                exit 1
            fi
            if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                err "-n must be a positive integer, got: '$2'"
                exit 1
            fi
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --full)
            USE_FULL=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        [1-9]*)
            # Positional shorthand: bash script/test_pipeline.sh 50
            if ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
                err "Positional argument must be a positive integer, got: '$1'"
                exit 1
            fi
            NUM_SAMPLES="$1"
            shift
            ;;
        *)
            err "Unknown argument: '$1'  (run with -h for help)"
            exit 1
            ;;
    esac
done

# ── Step 0: slice the dataset ──────────────────────────────────────────────────
banner "============================================================"
banner "   MOON-SQL  |  Dynamic Test Pipeline"
banner "============================================================"
echo ""

info "Step 0/2 — Preparing dataset slice..."

if [[ "$USE_FULL" == true ]]; then
    SLICER_ARGS="--full"
elif [[ -n "$NUM_SAMPLES" ]]; then
    SLICER_ARGS="-n $NUM_SAMPLES"
else
    warn "No -n or --full provided — defaulting to --full."
    SLICER_ARGS="--full"
fi

# Capture all output; the terminal KEY=VALUE lines carry the resolved paths.
SLICER_OUTPUT=$("$PYTHON" "$SLICER" $SLICER_ARGS)
echo "$SLICER_OUTPUT"
echo ""

# Parse the KEY=VALUE lines emitted by src/utils/slice_data.py
_parse_kv() {
    echo "$SLICER_OUTPUT" | grep "^${1}=" | tail -n1 | cut -d'=' -f2-
}

RESOLVED_DEV_PATH=$(_parse_kv "DEV_PATH")
RESOLVED_DATA_MODE=$(_parse_kv "DATA_MODE")

# Validate that we got non-empty values
if [[ -z "$RESOLVED_DEV_PATH" || -z "$RESOLVED_DATA_MODE" ]]; then
    err "slice_data.py did not emit expected KEY=VALUE lines."
    err "Output was:"
    echo "$SLICER_OUTPUT" >&2
    exit 1
fi

info "Resolved DEV_PATH  : $RESOLVED_DEV_PATH"
info "Resolved DATA_MODE : $RESOLVED_DATA_MODE"

# ── Step 1: clear stale cache, then generate ─────────────────────────────────
echo ""
banner "------------------------------------------------------------"
banner "   Step 1/2 — Generation  (run.sh)"
banner "------------------------------------------------------------"

# Remove all files from the two cache directories so that run.sh always
# starts from scratch instead of skipping already-present outputs.
# The directories themselves are preserved (run.sh expects them to exist).
for cache_dir in \
    "$PROJECT_ROOT/data/generate_datasets" \
    "$PROJECT_ROOT/data/intermediate_datasets"
do
    if [[ -d "$cache_dir" ]]; then
        file_count=$(find "$cache_dir" -maxdepth 1 -type f | wc -l)
        if [[ "$file_count" -gt 0 ]]; then
            find "$cache_dir" -maxdepth 1 -type f -delete
            info "Cleared $file_count cached file(s) from $cache_dir"
        else
            info "Cache already empty: $cache_dir"
        fi
    fi
done

export DEV_PATH="$RESOLVED_DEV_PATH"
export DATA_MODE="$RESOLVED_DATA_MODE"

bash "$SCRIPT_DIR/run.sh"

# ── Step 2: evaluation (eval.sh) ──────────────────────────────────────────────
echo ""
banner "------------------------------------------------------------"
banner "   Step 2/2 — Evaluation  (eval.sh)"
banner "------------------------------------------------------------"

# DEV_PATH and DATA_MODE are already exported; eval.sh reads them too.
bash "$SCRIPT_DIR/eval.sh"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
banner "============================================================"
banner "   Pipeline complete!"
banner "============================================================"
info "Data mode : $RESOLVED_DATA_MODE"
info "Input     : $RESOLVED_DEV_PATH"
