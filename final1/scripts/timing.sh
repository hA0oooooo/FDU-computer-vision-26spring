#!/usr/bin/env bash

TIMING_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMING_CSV="${TIMING_CSV:-${TIMING_PROJECT_ROOT}/report/timing.csv}"

init_timing_csv() {
  mkdir -p "$(dirname "$TIMING_CSV")"
  if [ ! -f "$TIMING_CSV" ]; then
    printf 'object,stage,elapsed_seconds,exit_status\n' > "$TIMING_CSV"
  fi
}

append_timing_row() {
  local object_name="$1"
  local stage_name="$2"
  local elapsed_seconds="$3"
  local status="$4"

  printf '%s,%s,%s,%s\n' "$object_name" "$stage_name" "$elapsed_seconds" "$status" >> "$TIMING_CSV"
}

run_timed() {
  local object_name="$1"
  local stage_name="$2"
  shift 2

  init_timing_csv

  local start_epoch
  local end_epoch
  local status

  start_epoch="$(date +%s)"

  set +e
  "$@"
  status=$?
  set -e

  end_epoch="$(date +%s)"

  append_timing_row "$object_name" "$stage_name" "$((end_epoch - start_epoch))" "$status"

  return "$status"
}
