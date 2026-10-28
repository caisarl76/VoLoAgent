# Commit history rewrite for DCO sign-off, 27 October 2026

CONTRIBUTING.md requires a `Signed-off-by` trailer, with a real name and email, on every commit. The review of VoLoAgent#5 (the voloagent-5a session) found 43 of the branch's commits without one. Like the 19 October rewrite ([`../g1_commit_rewrite_20261019`](../g1_commit_rewrite_20261019/README.md)), they were authored as `Jihun Kim <caisarl76>`, which has no domain.

The branch `work/g1-autonomous-loop-20261026` (49 commits on `fork/main`) was rewritten in place. The user chose the same identity as on 19 October.

| Commits | Change |
|---|---|
| 43 domain-less | author and committer email set to `caisarl76@gmail.com`, plus a matching sign-off |
| 2 already on `caisarl76@gmail.com` | a matching sign-off added |
| 4 by `jihun.kim@genon.ai` | unchanged: they already had a matching sign-off |

Every commit's tree, author name, author date, committer date, subject and message text are unchanged (checked commit by commit). Each commit now has exactly one sign-off matching its author. The final tree is byte-identical to the pre-rewrite head `ad63ab2`.

## Historical SHAs are kept

PR descriptions, the ledger, cross-session messages and the robot's `VOLO_SOURCE_COMMIT` files cite pre-rewrite SHAs, such as `44c84ab` (deployed for r41) and `ad63ab2`. Those identify the code that was actually tested, so they are not edited. The pre-rewrite history stays reachable through the tag `pre-dco-20261027` on the fork.

[`mapping.txt`](mapping.txt) lists, per commit, the old SHA, the new SHA, the original author email and the subject. Native GR00T-WholeBodyControl SHAs (PR #19, `f192d29`) are not affected.
