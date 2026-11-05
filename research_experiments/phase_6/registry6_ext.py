"""Append-only registry extensions (imported by registry6.py). Never remove entries."""
from research_experiments.phase_6.registry6 import M, REPEATS, register, runs

register("P6-stage1-screen", "A stage 1 (fine-tuning screen)", M + "validation.screen",
         artifacts=["results/stage1_screen.json", "results/stage1_screen.csv"])
register("P6-encoder-licence-gate", "B stage 1 (encoder screen)", "(licence check by hand; see configs/encoders.json)",
         artifacts=["configs/encoders.json"])
register("P6B0-ecgf-f6-v1", "B stage 1 (encoder screen)", M + "experiments.encoders_b --emb B0", runs("P6B0-ecgf-f6-v1", (42,), REPEATS))
register("P6B-ecgfm-embed", "B stage 1 (encoder screen)", M + "experiments.embed_ecgfm", artifacts=["results/ecgfm_embedding_validation.json"])
register("P6B-ecgfm-v1", "B stage 1 (encoder screen)", M + "experiments.encoders_b --emb ECGFM", runs("P6B-ecgfm-v1", (42,), REPEATS))
