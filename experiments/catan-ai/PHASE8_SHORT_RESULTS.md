# Phase 8 bounded diagnostic and throughput pass

Bounded phase8 inference-cache correctness/throughput and value/opponent diagnostics; full chance allocation sweep deferred.

No new games, training, Transformer or random-player runs. Twelve fixed validation positions; cache and baseline alternate measurement order at 500 ms, with a fixed-16-simulation correctness comparison.

```json
{
  "positions": 12,
  "brier": 0.3119940019219154,
  "mean_value_bce": 0.9050099651853876,
  "frozen_opponent_teacher_top1": 0.25,
  "baseline_median_simulations_per_second": 203.82789865356276,
  "cached_median_simulations_per_second": 244.42534844246015,
  "throughput_change_fraction": 0.19917513773666018,
  "fixed_simulation_equivalence_passed": true,
  "perspective_check": "Actor-to-root conversion verified on every sample; full regression suite covers private information and chance invariance",
  "limitations": "12 stratified validation positions; value calibration and frozen-opponent imitation are diagnostic only, not on-policy strength or opponent-response calibration.",
  "decision": "Retain C0/c=1.5 and existing runtime; optional cache is not installed. No search strength gain established, so defer C1-C4 and progressive widening.",
  "full_original_phase8_complete": false,
  "next_work": "Learner improvement/data collection is the next substantive stage; no automatic large generation queue."
}
```

The cache key includes permitted features, legal actions and development-card knowledge. Caches are separate per checkpoint and per decision and are not installed as a playing default. The original phase-eight chance-allocation/progressive-widening comparison is deferred, not claimed complete.
