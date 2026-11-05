# Phase 6 handoff (living document; updated at every milestone)

- **Branch:** `research/phase6`, created from `research/phase5` @ `27cbdbd`, pushed to `origin` (the `chaithr45`
  fork). No force-push; no other branch modified; never push to `claude/funny-gauss-udlmv0`.
- **Read first:** `PHASE6_RESUME.md` (commands), `PHASE6_PLAN.md` (pre-registration), `PHASE6_STATUS.md` (live
  status), `PHASE6_CLAIMS_REGISTER.md`, `REPOSITORY_AUDIT_P6.md`.

## State
**Handover (2026-11-05):** the remaining Phase 6A / 6B computation moves to the owner's laptop (NVIDIA GPU) in a local Claude Code session. Per-ECG predictions are private and machine-local, so that session regenerates every input and baseline deterministically (PHASE6_RESUME.md; baselines must be 24/24 byte-identical) and then runs everything still open. The cloud session stopped its LSTM runs (L4 29/45 folds, L1 1/45; nothing committed from them) to avoid computing anything twice.

| Milestone | State |
|---|---|
| M0 setup + pre-registration | DONE |
| M1 timing (experiment A) | DONE (plan Amendment 1) |
| A stage 1 | DONE: R0 reproduces Phase 4; A1 and A2 fail the screen (r 0.650 vs 0.652) |
| B stage 1 | DONE for ECG-FM (fails; also combined with ECGFounder); HuBERT-ECG / ST-MEM need an owner licence decision; ECG-JEPA / MERL blocked (Google Drive) |
| D, Stage 2, E, N, final report | not started |
| C (extra ECGs) | not run: blocked (no extra PDFs / waveforms; owner-confirmed) |

## Known issues (owner decisions; recorded, not acted on)
1. **Public exposure:** the public repository tracks the patient-named PDFs, its history contains the deleted
   `ECG_ECHO` sheets, and Phase 1–3 per-ECG CSVs (incl. `phase_2/locked_test/`) are tracked. The owner handles this;
   Phase 6 modifies none of it and adds nothing similar.
2. **Name pairs:** 29 near-identical name pairs under different keys: 15 within D (sensitivity analysis in Phase 6),
   **11 straddling D and the locked test**, 3 within the locked test. The 11 + 3 must be resolved before any
   locked-test evaluation.
3. **Locked test:** untouched in Phase 6; no test PDF is run through any model.
