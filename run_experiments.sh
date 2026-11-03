#!/usr/bin/env bash
# Full pipeline: preprocess -> classical baseline -> CNNs -> ensemble -> comparison table.
# On a 4-core CPU this takes ~4.5 hours; on a GPU a few minutes per model.
# Add --pretrained to the CNN lines when internet access to the timm/HF weights is available.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

[ -f data/manifest.csv ] || python -m ecg_ef.preprocess

run() { local tag=$1; shift; python -m ecg_ef.train "$@" 2>&1 | tee "logs/$tag.log"; }

python -m ecg_ef.baseline 2>&1 | tee logs/hog_svm.log
run ecgnet_filtered          --model ecgnet          --variant filtered
run ecgnet_raw               --model ecgnet          --variant raw
run resnet18_filtered        --model resnet18        --variant filtered
run efficientnet_b0_filtered --model efficientnet_b0 --variant filtered
run convnext_atto_filtered   --model convnext_atto   --variant filtered
run mobilenetv3_filtered     --model mobilenetv3     --variant filtered
run ecgnet_filtered_384      --model ecgnet          --variant filtered --img-size 384 768 --tag ecgnet_filtered_384
run convnext_atto_lr2e-4     --model convnext_atto   --variant filtered --lr 2e-4 --tag convnext_atto_filtered_lr2e-4

python -m ecg_ef.ensemble
python -m ecg_ef.compare
