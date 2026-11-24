# CNN policy cutoff investigation

The seed-11 XD-CNN policy finished its 100 tuning pairs (200 games): 43 wins,
153 losses, four cutoffs, no errors or illegal actions. Completed-game win rate
is 43/196 = 21.94%; match score including the predeclared half-credit for cutoffs
is 22.5%. All four cutoffs hit 1,000 turns, not the action or decision-time cap.
The 2% cutoff rate fails the frozen 1% promotion gate.

| Pair | CNN VP | SMART VP | Last build turn | Last development purchase turn |
| --- | ---: | ---: | ---: | ---: |
| tuning-0042 | 12 | 14 | 173 | 123 |
| tuning-0072 | 12 | 14 | 192 | 98 |
| tuning-0081 | 8 | 14 | 149 | 120 |
| tuning-0082 | 9 | 14 | 137 | 119 |

Every cutoff trajectory was replayed from logged actions and seeds; all four
SHA-256 values matched. At termination the development deck was empty, SMART
had all four cities and longest road, and the CNN held largest army. Neither
player had an immediately buildable settlement location. Both still had legal
road-extension opportunities, so these are observed behavioral stalls, not
proven mathematically unwinnable positions. For the last 100 turns they mostly
rolled, discarded, moved the robber, traded, and ended turns without expanding.
The reason SMART chooses not to extend roads here has not been isolated.

The gate is working: this policy result remains ineligible for promotion.
The original orchestration unnecessarily stopped the entire comparison on that
failure. Following investigation, the complete-comparison driver may retain an
ineligible cutoff result and continue other architectures/search modes. The
1% threshold, turn cap, cutoff scoring, checkpoints, decision code, native
SMART implementation and frozen seeds are unchanged. Search must meet its own
cutoff and timing gates. No cutoff is relabeled as a win or discarded.

The already completed CNN smoke and screening records are reused with explicit
compatibility provenance. Only the orchestration code hashes changed; the
resolved decision configurations, provider/native identities and log checksums
are verified before reuse. No completed CNN games are rerun.

Detailed terminal hands, piece counts, legal opportunities, final-turn action
counts and last building/purchase turns are in ignored
`runs/catan-ai/phase5-cnn-cutoff-diagnosis.json`. The replay script is retained at
`runs/catan-ai/cnn-cutoff-inspect.py`.
