#!/usr/bin/env bash
# Batch-render domain figures (fault traces + rupture propagation +
# hypocentre) for every realisation JSON, in parallel, skipping any
# already done and logging any that fail to load/plot. Resumable.
set -uo pipefail

REPO=/home/arr65/src/visualisation
SRC=/home/arr65/src/workflow/complete_realisations
OUTDIR="${1:-$HOME/complete_realisation_figures}"
JOBS="${2:-6}"
FLAGS="--show-traces --show-rupture-propagation --show-hypocentre"

mkdir -p "$OUTDIR"
: > "$OUTDIR/failures.log"

render_one() {
  local f="$1"
  local id out
  id=$(basename "$f" .json)
  out="$OUTDIR/$id.png"
  if [[ -s "$out" ]]; then
    echo "SKIP $id"
    return 0
  fi
  if "$REPO/.venv/bin/plot-domain" "$f" "$out" $FLAGS --title "$id" >/dev/null 2>&1; then
    echo "OK   $id"
  else
    echo "FAIL $id"
    echo "$id" >> "$OUTDIR/failures.log"
  fi
}
export -f render_one
export REPO OUTDIR FLAGS

total=$(ls "$SRC"/realisation_*.json | wc -l)
echo "Rendering $total realisations -> $OUTDIR  (jobs=$JOBS, flags=$FLAGS)"
ls "$SRC"/realisation_*.json | xargs -P "$JOBS" -I{} bash -c 'render_one "$@"' _ {}

done_n=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
fail_n=$(wc -l < "$OUTDIR/failures.log")
echo "--------------------------------------------------"
echo "Done. $done_n PNGs in $OUTDIR ; $fail_n failed (see $OUTDIR/failures.log)"
