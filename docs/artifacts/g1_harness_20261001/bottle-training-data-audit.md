# Failed episode retained in bottle training data

The user-labeled unrecovered failure in bottle episode 5 is retained in the
prepared training corpus. This warrants an outcome audit before further VLA
fine-tuning. It does not establish that the current checkpoint learned to fail
or that this episode caused a particular inference failure. SONIC turning is
a separate controller-tracking issue.

The user's output video and immutable raw-source episode 5 have identical
SHA-256:
`24d59c1358e8a95b7ad5b66880a879499b18183a72935cec2561612836d77445`.
The prepared episode-5 video also has that hash. Raw episode metadata says
`place OBJECT on the table`, length 977. The source mapping retains source
episode 5 as prepared episode 5, also length 977, relabeled
`pick drink bottle and place it on the right table`.

The experiment's [README](/home/jihun/work/GR00T-WholeBodyControl/docs/artifacts/pnp_bottle_260916_raw_training_20260930/README.md:5)
records 179 raw episodes, dropping only 13, 94, 99, 106 and 109, yielding
174 episodes / 274,241 frames, and training to checkpoint 20000.
[Preparation code](/home/jihun/work/GR00T-WholeBodyControl/docs/artifacts/pnp_bottle_260916_raw_training_20260930/prepare_dataset.py:63)
copies each retained video and assigns the same task prompt. These records
support retention in the declared training corpus; no per-step sampler trace
was inspected to prove exactly which windows were consumed by the optimizer.

[Audit evidence](bottle-training-data-audit.json) records source paths, video
hashes, metadata and mapping. [Failed-episode monitoring](vision-bottle-failure-genon/README.md)
records four incomplete samples without false success. No source dataset,
training label or checkpoint was modified.

Before another training run:

1. Label episode outcomes: successful placement, failed without recovery,
   failed then recovered, and outcome unclear.
2. Decide whether failure segments belong in demonstrations; retain useful
   recovery behavior with explicit segment boundaries and task labels.
3. Rebuild a versioned dataset and a separate evaluation set, with provenance
   linking source episodes to retained training segments.
4. Train additional subgoal prompts only when their demonstrations and
   evaluations exist. Registering a prompt alone teaches no new robot behavior.
