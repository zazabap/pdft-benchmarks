#!/usr/bin/env bash
# Re-derive every canonical cell from a fresh checkout.
#
# Wall clock on 2x RTX 3090 (concurrent): ~3 hours.
# - DIV2K-10q block bases:  ~50 min/basis × 3 = 2.5 h on GPU 0
# - QuickDraw all 6 active: ~5 min/basis × 6  = 30 min on GPU 1
# - DIV2K-8q (already in archive) is not retrained — extracted only.
#
# Usage:
#     bash scripts/run_canonical.sh
#     GPU0=0 GPU1=1 bash scripts/run_canonical.sh
#
# Set ONLY_EXTRACT=1 to skip training and just (re-)build results/published/
# from whatever already exists in results/_archive/.
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

if [ -z "${ONLY_EXTRACT:-}" ]; then
    echo "[run_canonical] starting div2k_10q_block on GPU ${GPU0} (background)…"
    python experiments/div2k_10q_block.py --gpu "${GPU0}" &
    pid_10q=$!

    echo "[run_canonical] starting quickdraw on GPU ${GPU1} (background)…"
    python experiments/quickdraw.py --gpu "${GPU1}" &
    pid_qd=$!

    wait "${pid_10q}" "${pid_qd}"
    echo "[run_canonical] both training jobs finished."
    echo "[run_canonical] NOTE: now move new results dirs into results/_archive/"
    echo "                and update EXTRACTION_TABLE in scripts/extract_canonical_cells.py"
    echo "                to point at them, then re-run with ONLY_EXTRACT=1."
    exit 0
fi

echo "[run_canonical] extracting cells…"
python scripts/extract_canonical_cells.py

echo "[run_canonical] (re)building MANIFEST.json + README.md…"
python scripts/render_published_readme.py

echo "[run_canonical] validating…"
python scripts/validate_manifest.py

echo "[run_canonical] OK"
