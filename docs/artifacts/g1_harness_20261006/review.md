# Fresh review and executor rulings — 2026-10-06

A fresh-context `gpt-6-astra` reviewer with high reasoning inspected the native
observation fix, its test, and the current authored experiment scripts against
VoLoAgent base `ce6cf58d1f2c484d6a8866e4c246ddfcb74cc2ec` and native base
`73560de922f27af97b7cdeaf41bba43c0f809feb`. Review was read-only; no GPU,
network, DDS or physical actuation was used. The main agent owns final
documentation, packaging and integration checks.

The reviewer independently ran 24 passing focused tests: one native observation
test, 22 placement tests and one mocked mission failure-path test. The final
verdict was ready to commit for the reviewed code scope, with no unresolved
critical or important findings.

Two important findings were fixed:

- Wall time alone allowed a purported 0.5-second dwell with only 5 milliseconds
  of simulated physics. The window now spans at least 0.5 seconds in both clocks;
  it walks back far enough to allow modestly slower physics. A counterexample
  failed before the fix and passes after it. Nonfinite destination bounds also
  fail closed.
- A coordinator completion followed by an evidence-read exception could retain
  `outcome="completed"`. The outcome now stays failed until coordinator and
  independent placement checks both pass. The mocked completed-coordinator /
  missing-truth case reproduces the old defect using the archived driver and
  passes with current code. No subprocesses or robot sockets execute in it.

The minor diagnostic naming issue was fixed by renaming the current output to
`destination_contact_observed`. Historical results keep their original keys and
hashes. Scripted truth controls no longer require a Genon API key; actual Genon
mode still uses the user's supplied endpoint and credential source.

The main agent accepts each behavior the reviewer declined to judge:

| Behavior outside scoped review | Executor ruling |
| --- | --- |
| Physical deployment safety or success | Unaccepted; no physical actuation occurred. |
| Policy generalization | Unestablished; replay is in-training and live bottle trials failed. |
| Full scene calibration | Partial bottle measurements only; camera/layout/contact assumptions remain. |
| Dataset outcome labels or training improvements | Numeric candidates only; no relabeling, filtering or training. |
| Historical as-run implementations | Preserve exact snapshots; current code carries the fixes. |
| General-purpose portability | Dated workstation experiment scripts with explicit local dependencies. |
| README, provenance and turning plots | Root checks raw counts, renders figures, checks links and hashes. |
| Full-suite results | Root inspected successful logs: native627, VoLo595/15skips, required G1 50/0skips. |

The root also checked the fresh turn evidence: two failed-turn probes exit before
post-interruption hold acknowledgement. Their hold flag is false at the recorded
instant. The report does not claim confirmation for that path; it remains a
follow-up alongside turn repeatability and actual-native-loop locomotion.
