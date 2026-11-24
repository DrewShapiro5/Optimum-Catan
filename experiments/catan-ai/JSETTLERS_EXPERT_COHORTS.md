# Additional JSettlers expert cohorts — November 18, 2026

User requested a three-hour expert-training window after the current fresh W4
CNN scored 2 wins / 8 losses against JSettlers at 500 ms search. The previous
12-worker self-play collector was gracefully paused with 62 completed games
and 12 interrupted prefixes. Its source files, plan, checkpoints and journals
remain unchanged and available for a later explicit resume.

Run/resume: `.conda/bin/python -m agent_apis.neural.expert_cohorts --hours 3`.
Progress: `runs/catan-ai/jsettlers-expert-cohorts-visible/progress.json`.
Log: `runs/catan-ai/jsettlers-expert-cohorts-visible.log`.

Continue the corrected fresh lineage from
`runs/catan-ai/fresh-xdcnn-visible/pretraining/iteration-1/best.pt`, checksum
`8ab61951e5bb0fcae6f022d5a08f191d52cee753cf17274aee0bac517bc08a18`.
This is continuation of the already fresh-initialized model, not a new random
restart or an import of historical incorrect-rule weights.

Each iteration generates 2,000 NEW completed JSettlers-versus-JSettlers games
with 12 native game workers and one PyTorch CPU thread per worker. There is no
CNN search in expert collection. Friendly robber uses visible victory points.
A durable completed-game slot fixes the split before dispatch; cutoff retries
keep that slot and get fresh seeds. Each cohort has 1,800 train / 100 validation /
100 test games. No updates occur before the full 2,000-game cohort completes.
Test games never enter training or checkpoint selection.

Keep compressed action/chance journals, seeds and checksums, with exact replay
verification. Resume reuses completed games; interrupted prefixes are archived
before restarting their seed. Derive at most 64 seeded positions per game in
RAM. Both independent halves of the pinned trainer sample the current expert
training pool, making the update expert-only. Use unchanged W4 half-uniform,
half-capped-family sampling and value correction exactly once. The existing
2,000-update AdamW block and MPS optimizer remain unchanged. Validate every
250 updates including update zero; continue from the earliest best validation
checkpoint and preserve final weights plus optimizer/RNG recovery state.

Each new cohort gets its own manifest and training folder. The pinned trainer's
historical metadata labels include `selfplay` even when both input pools are
expert data; the execution protocol and manifests explicitly identify this
expert-only route. The new source archive preserves the exact implementation.
No automated deployment, self-play resume, or window extension is performed.
At three hours stop at a safe action/update boundary and preserve incomplete
cohorts without training on a small prefix.
