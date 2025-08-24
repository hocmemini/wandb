#!/usr/bin/env bash
set -euo pipefail
echo "Window A: wandb launch-agent --queue cv-queue"
echo "Window B: wandb launch --queue cv-queue --project yolo-demo --entity <your_team> --entry-point train.py -- --project yolo-demo --epochs 3"
