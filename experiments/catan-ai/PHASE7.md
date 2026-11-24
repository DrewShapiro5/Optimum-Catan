Current execution amendment (November 12, 2026): phase 7 closed early at the
user request; remaining simulation-budget and confirmation matches are abandoned.
Use existing XD-CNN D2/W0 results, retain c=1.5/C0 provisionally, and do not
claim a demonstrated search gain. Phase 8 now uses a bounded 12-position
correctness/throughput and value/opponent diagnostic; the exhaustive chance
allocation sweep is deferred. No Transformer or random-player runs, no automatic
large match or self-play queue. See [PHASE7_EARLY_RESULTS.md](PHASE7_EARLY_RESULTS.md) for the preserved interim evidence.

Opponent amendment (November 11, 2026): **JSettlers only** from now on.
The user removed random-player testing. Completed JSettlers games are reused;
random-player history is retained but excluded from active scheduling and gates.
This halves phase-seven new-game limits to 400 screening games plus at most
100 additional confirmation games. Search settings and training are unchanged.

# Phase 7: XD-CNN search strength and latency

Authorized November 11, 2026. XD-CNN is the only candidate architecture. Use the
phase-six conservative D2/W0 recipe and existing validation-selected checkpoints
at training seeds 11, 23 and 47. No new training, Transformer runs, or final
confirmation seeds are launched. The versioned protocol is [phase7.json](phase7.json).

Reuse the phase-six pure-policy and 500 ms/c=1.5 results for each training seed
and each opponent. Verify model checkpoints, decision implementations, frozen
opponent identities, configuration hashes and game/timing checksums. Preserve
the original report and source identities. No historical games are replayed,
pooled with a new revision, or represented as newly instrumented measurements.

At training seed 11, compare c_puct=0.75 and 3 against the reused c=1.5 reference
at 500 ms. Also measure c=1.5 at 100 and 2,000 ms, and at exactly 128, 512 and
2,048 total simulations per non-forced decision across four root workers. The
heuristic control uses the same neural legal-action priors, frozen seed-11 MLP
opponent and search, replacing only leaf win probability with the existing
observation-only score/resource logistic evaluator. This still evaluates the
neural network for its priors; it is a value-quality control, not a claim about
speed without neural inference.

Each new variant plays both JSettlers SMART and random player: 10-game smoke
followed by at most 50 total games, with smoke included. Both opponents' smoke
runs precede extensions within that variant. The 500 ms exploration sweep runs
first; fixed-simulation diagnostics run last. Preserve the existing futility,
cutoff, error and paired-seat rules. Fixed-simulation experiments are not
wall-clock constrained: report their actual latency; the raw benchmark's 500 ms
flag does not determine their diagnostic validity. They cannot qualify for the
primary 500 ms promotion gate.

Choose the highest eligible seed-11 500 ms neural-search JSettlers score, with
c=1.5 preferred on ties, then .75, then 3. Require random-opponent search gates
as well. Evaluate a newly selected constant at training seeds 23 and 47; reuse
existing seed23/47 evidence if c=1.5 wins. Search promotion requires an eligible
three-seed search comparison and a positive paired hierarchical 95% lower
bound against its own policy. Preserve and disclose existing policy cutoffs.
Any optionally stopped comparison is descriptive and cannot promote search.
If search fails this gate, diagnose calibration, perspective and opponent-model
mismatch before adding complexity. No installed default changes automatically.

The instrumented provider inherits the verified stochastic decision, chance,
hidden-world and backup implementation. It retains the optimized four-worker
batching, named random streams and admission rule. Tests compare exact serial
root statistics before/after instrumentation at fixed simulations. The provider
records inference requests/batches/service time, observation and chance cache
hits, transition/observation/evaluation time, raw versus merged chance branching,
completed-turn depth, root visit entropy and simulation throughput. Neural
output caching is absent and its hit count is explicitly zero. Subsystem times
can overlap and sum across threads; inference service includes packing,
encoding, transfer and compute. They must not be added as wall-clock fractions.
Historical references expose only their original telemetry subset.

There are at most 800 new seed-11 games and 200 additional games if another
exploration constant requires three-seed confirmation. The high-simulation
and 2,000 ms diagnostics can take substantially longer than 500 ms games.

Run or resume with `.conda/bin/python -m agent_apis.neural.phase7`.
The queue holds the exclusive benchmark lock, persists state/configurations,
verifies compatible artifacts before reuse, and uses the checksum-committed
per-game resume path. Errors stop for investigation with no fallback opponent.

Live status: `runs/catan-ai/phase7-progress.json`.
Live log: `runs/catan-ai/phase7.log`.
Interim table: [PHASE7_PROGRESS.md](PHASE7_PROGRESS.md).
Final reports are written only after completion as `PHASE7_RESULTS.md` and `.json`.
