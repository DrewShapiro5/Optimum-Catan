# Subtree reuse results

Twelve frozen validation starts were advanced using independently sampled real chance outcomes and native JSettlers opponent decisions. No training, random player, long match sweep or final-seed evaluation was launched.

- Reuse succeeded on **10/12** real transitions.
- **85** retained root worlds matched fresh decoding exactly.
- All returned actions were legal; warm p95 **483 ms**, maximum **493 ms**.
- Median paired new-simulation throughput change: **-4.4%**; descriptive 95% bootstrap interval **[-21.4%, +4.5%]**.
- Ratio of unpaired workload median throughputs: **-18.5%**. This differs from the paired metric and must not be presented as a uniform speedup.
- Some re-rooted searches go deeper and perform fewer new simulations; retained visits are reported separately. No playing-strength improvement is established.
- Reuse remains **opt-in** via `search.reuse_subtree=true`; no further self-play training was started.

Compatible states, chance classes, observations and keys are reused; unrelated branches are released. The hidden-card posterior refreshes at every real decision. When opponent purchases introduce unknown card worlds, old visits/returns are reset while compatible state objects and priors remain reusable. Visits also reset when old simulations were capped or depth-limited. Known-world uncapped branches may keep their search statistics.

Fifty-seven regression tests passed, including real-checkpoint runtime routing, session/model revision isolation, hidden allocations, pruning, cache limits and fresh-decode avoidance. The full paired measurements and exact source hashes are in [SUBTREE_REUSE_RESULTS.json](SUBTREE_REUSE_RESULTS.json). See [the implementation contract](SUBTREE_REUSE.md) for activation and model-update rules.
