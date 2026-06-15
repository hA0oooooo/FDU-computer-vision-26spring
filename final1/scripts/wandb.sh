#!/usr/bin/env bash

WANDB_ENTITY="${WANDB_ENTITY:-fudan-university-CS50028}"
WANDB_PROJECT="${WANDB_PROJECT:-final-project}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/report/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/config}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${WANDB_DIR}/artifacts}"

wandb_enabled() {
  [ "$WANDB_ENABLE" = "1" ] || [ "$WANDB_ENABLE" = "true" ]
}

setup_wandb_env() {
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_ARTIFACT_DIR"
  export WANDB_ENTITY
  export WANDB_PROJECT
  export WANDB_MODE
  export WANDB_DIR
  export WANDB_CACHE_DIR
  export WANDB_CONFIG_DIR
  export WANDB_ARTIFACT_DIR
}

log_tensorboard_to_wandb() {
  local run_name="$1"
  local logdir="$2"

  if ! wandb_enabled; then
    return
  fi

  setup_wandb_env

  if [ ! -d "$logdir" ]; then
    echo "WandB TensorBoard log directory not found, skipping: $logdir" >&2
    return
  fi

  python "${PROJECT_ROOT}/scripts/log_tensorboard_to_wandb.py" \
    --logdir "$logdir" \
    --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" \
    --run-name "$run_name" || \
    echo "WandB TensorBoard upload failed for $run_name; training artifacts are unchanged." >&2
}
