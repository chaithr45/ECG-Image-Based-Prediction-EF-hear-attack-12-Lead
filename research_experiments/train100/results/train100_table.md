**Development cross-validation** (5 folds × 3 repeats unless stated)

| Model | Training accuracy (TRAIN100) | Held-out accuracy (TRAIN100) [corrected 95 % CI] | Held-out accuracy (original) | Change |
|---|---|---|---|---|
| HDR-LR | 52.9% | 48.3% [43.2%, 53.5%] | 48.9% | -0.6 points |
| HDRCLIN-LR | 58.9% | 48.6% [43.6%, 53.6%] | 50.1% | -1.5 points |
| CLIN-LR | 72.0% | 44.2% [39.0%, 49.3%] | 50.9% | -6.7 points |
| LGBM-FEAT | 100.0% | 54.8% [51.2%, 58.3%] | 52.4% | +2.4 points |
| ECGF-linear | 100.0% | 49.1% [45.1%, 53.2%] | 53.0% | -3.9 points |
| ECGF-MLP | 100.0% | 53.6% [50.6%, 56.5%] | 54.2% | -0.6 points |
| FUSION-MLP | 100.0% | 53.8% [49.9%, 57.8%] | 54.4% | -0.6 points |
| C1 | 100.0% | 51.7% [47.7%, 55.8%] | 54.7% | -3.0 points |
| C3 | 99.5% | 53.3% [49.6%, 56.9%] | 56.1% | -2.8 points |
| IMG-vote | 100.0% | 50.3% [46.3%, 54.4%] | 54.1% | -3.8 points |
| IMG-C3 | 100.0% | 53.2% [49.2%, 57.2%] | 55.8% | -2.6 points |
| AGT-img triage | 100.0% | 69.6% [63.0%, 76.2%] — answers 48% of held-out patients | 69.4% | +0.2 points |

**Original fixed split** (trained on 669 ECGs, measured on 168 validation ECGs)

| Model | Epochs | Training accuracy (TRAIN100) | Validation accuracy (TRAIN100) | Original: training / validation accuracy | Change |
|---|---|---|---|---|---|
