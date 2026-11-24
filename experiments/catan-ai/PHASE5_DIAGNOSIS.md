# Phase 5: severe MLP losses and evaluation stop

Evaluation was stopped at the user's request. All four seed-11 training runs
completed 20,000 updates, but **only the MLP has been evaluated against SMART**.
These results do not establish that CNN, GNN or Transformer also lose. Phase 5
remains incomplete; no workers or confirmation runs are running.

The completed 200-game policy screening and interrupted 87-game search screening
were replayed from their recorded board/simulator seeds and action IDs. Every
one of the 287 completed trajectories matched its recorded SHA-256 and winner.
This recovers terminal scores without rerunning agents or choosing new games.
Smoke games are already included in screening logs and were not counted again.
The interrupted search has 43 complete board pairs plus one completed seat;
the unfinished next game is excluded.

| Observed result | MLP policy | MLP 500 ms search |
| --- | ---: | ---: |
| Completed games | 200 | 87 |
| Wins / losses | 0 / 200 | 0 / 87 |
| Errors / cutoffs | 0 / 0 | 0 / 0 |
| Mean candidate VP | 3.68 | 4.03 |
| Mean SMART VP | 15.02 | 15.00 |
| Mean candidate margin | −11.34 | −10.97 |
| Median candidate margin | −12 | −11 |
| Settlements built after setup per game | 0.080 | 0.138 |
| City upgrades per game | 0.540 | 0.747 |

The unequal full-sample averages do not show search improvement. On the **same
43 complete pairs / 86 games**, candidate VP averages 4.081 for policy and 4.035
for search; search's mean margin change is −0.023 VP. Both have zero wins.
The 87 search games consumed approximately 93 minutes in total. These are
descriptive tuning results, not confirmation or a causal ablation.

## What fails

**Economic expansion stalls.** MLP policy makes 2.08 settlement placements per
game including two free setup placements; SMART makes 5.425. MLP upgrades 0.54
cities versus SMART's 3.615, and places 2.74 roads including setup versus 7.99.
Search scarcely changes the expansion pattern. In 56/200 policy games the MLP
finishes with only its initial two points. Opening production is also weaker:
0.482 expected resource cards per dice roll versus SMART's 0.586 (about 18%
lower, before robber effects). This is not mainly an inability to execute an
affordable city: policy ends only 78 of 11,786 turns with a settlement/city
immediately legal. The larger problem is reaching useful development states.

**Headline imitation accuracy hides weak strategic actions.** The selected MLP
is update 2,000, selected by the predeclared validation NLL+BCE criterion. Its
held-out policy accuracy is 63.0%, but road targets are matched only 17.1%,
settlement targets 32.3%, trades 24.8%, robber targets 26.1%, and knight plays
7.6%. In contrast, roll targets are matched 98.2% and end-turn targets 91.1%.
These are conditional target-family imitation metrics, not direct measures of
optimal action quality. They explain why aggregate accuracy is inadequate,
and support underfitting of important placement/development choices.

**Search has a weak learned evaluator and opponent model.** A fixed descriptive
sample of the first 20 logged games per mode was replayed for additional model
predictions. In search games, the MLP assigns a mean 67.1% win probability at its
first end-turn state, although all 20 games were losses. At its 20th end-turn it
still predicts 53.5%, with mean VP 2.8 versus 3.65. This is evidence of optimism
on its own search-selected states; it is not a full calibration estimate.
On actual SMART decisions sampled every tenth non-forced opponent decision,
the frozen MLP agrees on only 54.1% (policy trajectories, 218 samples) or 58.6%
(search trajectories, 198 samples) after excluding rolls. Search therefore plans
with a materially imperfect model of SMART as well as its own weak value/prior.
Its shallow rollout limit may also miss multi-turn expansion benefits.
These mechanisms are plausible explanations, not isolated causal findings.

No illegal actions, provider failures or trajectory mismatch explain these
losses. This does not rule out every modeling defect. Directly transferring
teacher-state imitation accuracy to self-play strength is the unvalidated step:
one bad move changes resource production and future states, so later mistakes
can compound outside the teacher's demonstration distribution.

## Changes and next work

Future tuning evaluations use the prospectively versioned [futility rule](FUTILITY.md).
A zero-win smoke stops at 40 games and prevents screening extension or automatic
confirmation training. Final score/margin fields are now stored directly.
Historical bootstrap [0,0] intervals are not evidence that the true win rate is
exactly zero, and new early-stopped runs suppress ordinary fixed-sample intervals.

Preserve the existing checkpoints and frozen seeds. Before expensive search or
three-seed confirmation, use small policy-only diagnostic samples for the other
trained architectures, inspect setup/expansion choices against SMART, and verify
whether later policy checkpoints trade imitation gains for worse value quality.
Then test action/phase imbalance, value calibration and opponent-model quality
as separate, documented experiments. No new matches or training were launched
as part of this diagnosis; evaluation remains stopped.

Machine-readable results and source/log hashes are in
[phase5-diagnosis-summary.json](phase5-diagnosis-summary.json). Exact per-game
recoveries and sampled predictions are in ignored
`runs/catan-ai/phase5-loss-diagnosis.json`.
