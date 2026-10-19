#!/bin/bash
# =============================================================================
# Google Cloud Batch task: run ONE line of jobs/jobs.tsv, Spot-preemption safe.
#   env STUDY_URI  gs://<bucket>/pellet-pd/v2/studies/<study>     (required)
#   env JOBS_FILE  jobs/jobs.tsv (default) - e.g. jobs/jobs_c4d-highcpu-32.tsv
#   env SYNC_EVERY seconds between output syncs to the bucket (default 600)
#   BATCH_TASK_INDEX is set by Cloud Batch (0-based); for manual use pass it as $1.
# Flow: compile solver for this CPU -> stage inputs -> pull any earlier attempt
# (checkpoint) -> run with --resume -> sync every SYNC_EVERY s and on SIGTERM ->
# on success delete the checkpoint and mirror the run directory to the bucket.
# Exit codes: 0 ok, 75 preempted (Batch retries via lifecyclePolicies), other = failure.
# =============================================================================
set -uo pipefail
: "${STUDY_URI:?STUDY_URI must be set, e.g. gs://BUCKET/pellet-pd/v2/studies/campaign-A}"
IDX=${BATCH_TASK_INDEX:-${1:-0}}
JOBS_FILE=${JOBS_FILE:-jobs/jobs.tsv}
SYNC_EVERY=${SYNC_EVERY:-600}
WORK=${WORK:-/tmp/pd}
mkdir -p "$WORK" && cd "$WORK" || exit 1

# 1) solver compiled natively for this VM (C3D Genoa / C4D Turin / anything) - cached per VM
if [ ! -x bin/pellet_pd ]; then
  mkdir -p bin
  BUILD=$(sha256sum /opt/pellet/pellet_pd.cpp | cut -c1-16)
  g++ -O3 -march=native -fopenmp -std=c++17 -DPD_BUILD="\"$BUILD\"" /opt/pellet/pellet_pd.cpp -o bin/pellet_pd || exit 1
fi
export PELLET_PD="$WORK/bin/pellet_pd"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

# 2) this task's line
mkdir -p jobs histories inputs runs
gcloud storage cp "$STUDY_URI/$JOBS_FILE" "$JOBS_FILE" --quiet || exit 1
LINE=$(sed -n "$((IDX + 1))p" "$JOBS_FILE")
CASE=$(printf '%s' "$LINE" | cut -f1)
CMD=$(printf '%s' "$LINE" | cut -f2-)
[ -n "$CASE" ] || { echo "no task $IDX in $JOBS_FILE"; exit 0; }
echo "task $IDX -> case $CASE on $(hostname) with $OMP_NUM_THREADS threads"

gcloud storage cp "$STUDY_URI/histories/$CASE.txt" histories/ --quiet 2>/dev/null || true
gcloud storage rsync -r "$STUDY_URI/inputs" inputs --quiet 2>/dev/null || true

# 3) earlier attempt (preempted) -> resume from its checkpoint
mkdir -p "runs/$CASE"
gcloud storage rsync -r "$STUDY_URI/runs/$CASE" "runs/$CASE" --quiet 2>/dev/null || true
if [ -f "runs/$CASE/summary.json" ] && grep -q '"complete": true' "runs/$CASE/summary.json"; then
  echo "case $CASE already complete"; exit 0
fi

sync_up() { gcloud storage rsync -r "runs/$CASE" "$STUDY_URI/runs/$CASE" --quiet; }
( while sleep "$SYNC_EVERY"; do sync_up; done ) &
SYNCPID=$!

setsid bash -c "$CMD" &          # own process group so SIGTERM reaches the solver
PID=$!
on_term() {
  kill -TERM -- "-$PID" 2>/dev/null
  wait "$PID"
  kill "$SYNCPID" 2>/dev/null
  sync_up                          # Spot gives ~30 s: push the latest checkpoint
  exit 75
}
trap on_term TERM INT
wait "$PID"
RC=$?
kill "$SYNCPID" 2>/dev/null
if [ "$RC" -eq 0 ]; then
  rm -f "runs/$CASE/checkpoint.bin" "runs/$CASE/checkpoint.bin.tmp"
  gcloud storage rsync -r --delete-unmatched-destination-objects "runs/$CASE" "$STUDY_URI/runs/$CASE" --quiet
else
  sync_up
fi
exit "$RC"
