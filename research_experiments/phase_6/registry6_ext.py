"""Append-only registry extensions (imported by registry6.py). Never remove entries."""
from research_experiments.phase_6.registry6 import M, register

register("P6-stage1-screen", "A stage 1 (fine-tuning screen)", M + "validation.screen",
         artifacts=["results/stage1_screen.json", "results/stage1_screen.csv"])
register("P6-encoder-licence-gate", "B stage 1 (encoder screen)", "(licence check by hand; see configs/encoders.json)",
         artifacts=["configs/encoders.json"])
