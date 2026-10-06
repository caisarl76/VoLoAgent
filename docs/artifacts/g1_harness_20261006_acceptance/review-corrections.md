# Review corrections — 2026-10-06

The review of commit `51d4d090044dda1f5bbf462f48d2cac0f64e150e` found two issues:
its missing author sign-off (P2), and list/object `split` values raising
`TypeError` in the dataset review checker (P3). Neither issue changes the saved
simulator or vision findings. No training bypass was found.

The checker now validates that a split is a string before set membership.
Two parameterized counterexamples first reproduced the exception, then passed
after the one-line fix. They require structured rejection, an invalid-split
error and no training windows, even when another episode has an otherwise
eligible training segment. Direct CLI checks also return JSON with exit 1 and
empty stderr for both malformed values.

The reviewed commit is amended with author identity
`Jihun Kim <caisarl76@gmail.com>` and its matching `Signed-off-by` trailer.
The email was verified against the authenticated GitHub profile and existing
native-repository author history; the prior VoLoAgent author email was only
`caisarl76`. This uses a command-specific email override and leaves persistent
Git configuration unchanged. Only the reviewed tip commit is replaced, using
an exact remote lease against its original SHA.

Fresh verification:

- [Red regression log](review-fix-split-red.log): two reproduced `TypeError`
  failures, eight other dataset tests deselected.
- [Focused checks](review-fix-checks.log): **29 passed**, no skips.
- [CLI counterexamples](review-fix-cli.json): both reject with structured JSON
  and exit 1; no traceback.
- [Ruff](review-fix-ruff.log): all checks passed.
- [Full suite](review-fix-full-suite.log): **595 passed, 15 skipped, 46 warnings**
  in 14.32 seconds. This command covers the project's `tests/`; the artifact
  regression tests are covered by the focused run.
- [Initial sandbox attempt](review-fix-full-suite-sandbox-abort.log): exited
  134 during `mock_gr00t_server` in `tests/test_gr00t_protocol.py`, at
  `bind_to_random_port`. Nine earlier failure markers have no detailed summaries
  because the process aborted. The same full-suite command passed outside the
  sandbox, with zero failures. Deprecation warnings are preserved in that log.

The simulator traces, model responses, source reviews and their original logs
are unchanged. `review-before/` preserves the checker, tests and manifest from
the reviewed commit. Current archive hashes include the corrections. Earlier
evidence archives remain unchanged. Training, manipulation, turn repeatability
and hardware acceptance remain open.
