#!/usr/bin/env bash

TIMING_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMING_CSV="${TIMING_CSV:-${TIMING_PROJECT_ROOT}/report/timing.csv}"

init_timing_csv() {
  mkdir -p "$(dirname "$TIMING_CSV")"
  if [ ! -f "$TIMING_CSV" ]; then
    printf 'object,stage,start_time,end_time,elapsed_seconds,notes\n' > "$TIMING_CSV"
  fi
}

append_timing_row() {
  local object_name="$1"
  local stage_name="$2"
  local start_time="$3"
  local end_time="$4"
  local elapsed_seconds="$5"
  local notes="${6:-}"

  notes="${notes//\"/\"\"}"
  printf '%s,%s,%s,%s,%s,"%s"\n' \
    "$object_name" "$stage_name" "$start_time" "$end_time" "$elapsed_seconds" "$notes" \
    >> "$TIMING_CSV"
}

run_timed() {
  local object_name="$1"
  local stage_name="$2"
  local notes="$3"
  shift 3

  init_timing_csv

  local start_epoch
  local end_epoch
  local start_time
  local end_time
  local status

  start_epoch="$(date +%s)"
  start_time="$(date -Iseconds)"

  set +e
  "$@"
  status=$?
  set -e

  end_epoch="$(date +%s)"
  end_time="$(date -Iseconds)"

  append_timing_row \
    "$object_name" \
    "$stage_name" \
    "$start_time" \
    "$end_time" \
    "$((end_epoch - start_epoch))" \
    "${notes}; exit_status=${status}"

  return "$status"
}
