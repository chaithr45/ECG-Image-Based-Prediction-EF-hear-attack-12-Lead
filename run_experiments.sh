#!/usr/bin/env bash
# Full pipeline: preprocess -> classical baseline -> CNNs -> comparison table.
# On CPU this takes a few hours; on a GPU a few minutes per model.
# Add --pretrained to the CNN lines when internet access to the timm/HF weights is available.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

[ -f data/manifest.csv ] || python -m ecg_ef.preprocess

python -m ecg_ef.baseline                                   2>&1 | tee logs/hog_svm.log
python -m ecg_ef.train --model ecgnet          --variant filtered 2>&1 | tee logs/ecgnet_filtered.log
python -m ecg_ef.train --model ecgnet          --variant raw      2>&1 | tee logs/ecgnet_raw.log
python -m ecg_ef.train --model resnet18        --variant filtered 2>&1 | tee logs/resnet18_filtered.log
python -m ecg_ef.train --model efficientnet_b0 --variant filtered 2>&1 | tee logs/efficientnet_b0_filtered.log
python -m ecg_ef.train --model convnext_atto   --variant filtered 2>&1 | tee logs/convnext_atto_filtered.log
python -m ecg_ef.train --model mobilenetv3     --variant filtered 2>&1 | tee logs/mobilenetv3_filtered.log

python -m ecg_ef.compare
