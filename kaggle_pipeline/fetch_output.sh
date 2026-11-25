#!/usr/bin/env bash
# Wait for the Kaggle kernel, then pull its output next to the corpus.
set -u
KERNEL="${KERNEL:-mohar07/legal-amr-sat-spring-wiki}"
DEST="${DEST:-data/kaggle_runs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$DEST"

for i in $(seq 1 180); do
  line="$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)"
  echo "[$(date +%H:%M:%S)] $line"
  state="$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')"
  case "$state" in
    *complete*|*error*|*failed*|*cancel*|*not\ found*) break ;;
  esac
  sleep 60
done

kaggle kernels output "$KERNEL" -p "$DEST" -o 2>&1 | tail -5
echo "output in $DEST"
ls -lh "$DEST" 2>/dev/null
