# Commit history rewrite for DCO sign-off, 19 October 2026

CONTRIBUTING.md requires a `Signed-off-by` trailer, with a real name and email,
on every commit. The review of VoLoAgent#1 at `53ca900` found 16 commits without
one. Those 16 were authored as `Jihun Kim <caisarl76>`, which has no domain and
so no valid email.

The branch `work/g1-harness-20261001` was rewritten on top of `b4e6230`:

- **The 16 domain-less commits:** author changed to
  `Jihun Kim <caisarl76@gmail.com>`, the identity of the neighbouring commits
  from the same period, with a matching sign-off added.
- **All other commits:** they already had a sign-off matching their author and
  kept it. None got a second sign-off.
- **Verified unchanged:** every commit's tree, author name, author date,
  subject and message text. The new sign-off adds one blank line before its
  trailer.
- **Result:** 53 commits, each with exactly one sign-off matching its
  author. The final tree is byte-identical to the pre-rewrite head.

## Historical SHAs are kept

Evidence READMEs, PR replies and PR descriptions cite pre-rewrite VoLoAgent
SHAs, such as `bdfb308` in the review-fix rerun. Those hashes identify the code
that was actually tested, so they are not edited. The pre-rewrite history stays
reachable through the tag `pre-dco-20261019` on the fork. Use
[`mapping.txt`](mapping.txt) (old SHA, new SHA, original author email, subject)
to find a commit on the current branch. Native GR00T-WholeBodyControl SHAs are
not affected.

| Old | New | Subject |
| --- | --- | --- |
| `bff8266` | `e2705f5` | docs: record G1 harness design and implementation plan |
| `f44f598` | `f6d5845` | feat: define G1 harness profile and RPC contract |
| `62a34ad` | `3c3cc7c` | docs: specify G1 native RPC and ownership contract |
| `7739699` | `480d880` | feat: monitor G1 bottle placement with fresh-frame confirmation |
| `2177289` | `324aeb9` | feat: run supervised G1 missions through VoLoAgent |
| `bfebcac` | `f552343` | test: verify G1 bottle mission and document launch |
| `37bdd08` | `65cd64a` | feat: add bounded G1 skill sequences and runtime capability checks |
| `49043dc` | `3232125` | fix: validate reset arguments and cancel lost start acknowledgements |
| `d5b4aa7` | `be91395` | test: verify agent takeover never resumes policy |
| `2118ddc` | `0e37906` | docs: record G1 harness verification and remaining hardware gates |
| `b341370` | `c658be7` | test(g1): record SONIC failures and Genon manipulation checks |
| `d3ff2fe` | `7cf2eb6` | docs: record G1 tracking limits and bottle monitor validation |
| `d9fe414` | `8eede6c` | fix: clarify G1 bottle destination and record live validation |
| `ce6cf58` | `f948f57` | docs: visualize saved G1 harness validation results |
| `95c2c44` | `9c90eef` | docs: record G1 input fix and measured bottle validation |
| `537b1e3` | `8f51a1f` | Validate native-loop stops and audit bottle training labels |
| `78aeeda` | `690e0de` | docs(g1): add dataset review gate and acceptance evidence |
| `2eee64b` | `d2b8f70` | Record instrumented G1 turn and D435i factory calibration |
| `b2e7e15` | `bedee35` | Save instrumented SONIC simulation replay videos |
| `394a10b` | `4dba4a9` | Preserve grasp across manipulation subskills and prepare review data |
| `8b7ef47` | `6d44536` | Apply bottle episode cuts and placement failure labels |
| `b386434` | `a6cf593` | Define stationary bottle handover autopilot and inspect recordings |
| `f1f755b` | `a692a3a` | Record confirmed handover references and candidate ready pose |
| `468a16b` | `0fa1693` | feat(g1): define reviewed handover profile and ready RPC |
| `2500fe4` | `42aed75` | feat(g1): monitor handover phases and latch roll failures |
| `f115bf3` | `d62e17d` | feat(g1): repeat stationary bottle handover under one lease |
| `aba19c8` | `f5fca4c` | fix(g1): retain handover context and audit trained baseline |
| `742f820` | `7cc030b` | docs: verify handover checkpoint on h100 GPU 0 and fix transfer sources |
| `3009290` | `0e2d38e` | docs: validate handover references and visualize native ready return |
| `1d88759` | `c400a40` | docs: inspect handover lifts and validate staged ready prototype |
| `c751c8d` | `762de84` | Evaluate Qwen GPU 6 for handover monitoring |
| `9677982` | `dc08b1b` | Use measured hand posture for handover offer confirmation |
| `137e1d7` | `a50b5cb` | Require handover lift waypoint and record native tracking failure |
| `7bff810` | `2180f35` | docs(g1): record paired tracking diagnosis and corrected simulation |
| `7612ba9` | `5a19039` | docs(g1): hand off ready-return tracking investigation |
| `7ccc26e` | `cf8ff71` | docs(g1): attribute ready-return stall to shoulder-yaw gravity load |
| `4e95b89` | `576ca0a` | docs(g1): screen raised waypoints by path yaw load and left-arm drift |
| `476d30c` | `cfa85c9` | feat(g1): accept handover desk pose and record direct ready returns |
| `5cc64aa` | `b5ab98c` | docs(g1): record post-merge ready-return trial from the ep15 offer |
| `7a762c3` | `a775736` | docs(g1): record post-merge ready-return trials from all three entries |
| `c6de9b4` | `dabf089` | feat(g1): mirror opt-in ready gravity feedforward; record yaw-tau trials |
| `4659ecc` | `610f373` | docs(g1): record accepted Dex3 smoke replay on the merged native branch |
| `94d258c` | `7a1809c` | feat(g1): set the handover desk edge to the measured 0.15 m |
| `e836c14` | `b5ee092` | docs(g1): record ready-return trials at the measured 0.15 m desk edge |
| `d02ea88` | `ac01a34` | docs(g1): record direct ready returns at the 0.15 m desk after the raise rule |
| `b6cdb9c` | `a20fa43` | feat(g1): configure the handover desk top above the floor |
| `1fe5045` | `f254ae9` | docs(g1): record ready-return trials with the online pelvis estimate |
| `70123c0` | `cd57cd6` | docs(g1): add ready-return replay videos |
| `73b3e26` | `cb74a56` | feat(g1): send per-session heartbeat sequences (RPC v1) |
| `bdfb308` | `eb916cf` | fix(g1): reset the handover monitor between autopilot sessions |
| `85ee923` | `32972c4` | docs(g1): record offer-trial rerun after the review fixes |
| `53ca900` | `814280b` | docs(g1): check where real-robot offers end against the measured desk |
| `5c3a8e7` | `5e3c411` | fix(g1): isolate monitor checks between missions; stop ready returns on camera loss |
