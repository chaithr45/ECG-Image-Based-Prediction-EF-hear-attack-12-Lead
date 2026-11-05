# Phase 6 handoff (living document; updated at every milestone)

- **Branch:** `research/phase6`, created from `research/phase5` @ `27cbdbd`, pushed to `origin` (the `chaithr45`
  fork). No force-push; no other branch modified; never push to `claude/funny-gauss-udlmv0`.
- **Read first:** `PHASE6_RESUME.md` (commands), `PHASE6_PLAN.md` (pre-registration), `PHASE6_STATUS.md` (live
  status), `PHASE6_CLAIMS_REGISTER.md`, `REPOSITORY_AUDIT_P6.md`.

## State
| Milestone | State |
|---|---|
| M0 setup + pre-registration | DONE |
| M1 timing (experiment A) | DONE (plan Amendment 1) |
| A stage 1 | next: activation cache → R0 check → A1 → A2 |
| B, D, Stage 2, E, N, final report | not started |
| C (extra ECGs) | not run: blocked (no extra PDFs / waveforms; owner-confirmed) |

## Known issues (owner decisions; recorded, not acted on)
1. **Public exposure:** the public repository tracks the patient-named PDFs, its history contains the deleted
   `ECG_ECHO` sheets, and Phase 1–3 per-ECG CSVs (incl. `phase_2/locked_test/`) are tracked. The owner handles this;
   Phase 6 modifies none of it and adds nothing similar.
2. **Name pairs:** 29 near-identical name pairs under different keys: 15 within D (sensitivity analysis in Phase 6),
   **11 straddling D and the locked test**, 3 within the locked test. The 11 + 3 must be resolved before any
   locked-test evaluation.
3. **Locked test:** untouched in Phase 6; no test PDF is run through any model.
