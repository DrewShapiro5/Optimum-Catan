# Visible-point friendly robber and private human-vs-bot display

User-requested fixes, November 17, 2026.

Friendly robber now protects enemies with fewer than three **visible** victory
points. Hidden victory-point cards never remove protection. Search beliefs no
longer use the legal robber mask or old robber placements to infer hidden VP
holdings. Engine, sampled-world reconstruction and belief tests cover identical
legal moves with 0, 1 and 3 hidden VP cards at two visible points.

For human-versus-bot games, GET state and POST action responses are pinned to the
human seat even on bot turns. Browser responses remove bot resource/development
identities, private hand flags, actual VP, private purchase/steal results,
remaining deck composition, and bot-turn legal-action menus. Requests for the
bot perspective in such games are rejected. The UI renders the human hand while
the bot searches and uses public resource/development totals for opponent card
backs. Existing human-only hotseat behavior remains available. The neural agent
still receives its own private observation directly from the engine, independent
of the browser projection. Saved games are preserved; refresh the app to load
the fixed UI. The Catan server remains at http://127.0.0.1:8005/.

Training resumes under `runs/catan-ai/fresh-xdcnn-visible/`, progress at
`runs/catan-ai/fresh-xdcnn-visible-progress.json`. The previous corpus/checkpoints
and source archive are immutable historical results for the old actual-point
variant. Every original completed expert journal is revalidated under the fixed
engine. Only legal, exactly replayable games are reused, with original splits
preserved and their original source references retained. Rejected games remain
archived and are replaced with new native games until train/validation/test
cohorts again total 1,800/100/100. Revalidation is an integrity audit, not test
metric evaluation; held-out test rows do not enter training or selection.

A fresh XD-CNN is trained only after the corrected 2,000-game cohort is complete.
Use W4, seeded up-to-64-position samples per game, best-validation checkpoint
selection, compressed reproducible journals and the existing bounded window.
The earlier missing-parent-folder setup error is also fixed permanently.

The user's observations of frequent development purchases, road overbuilding
and occasional poor setup are recorded as diagnostic evidence, without changing
the learned policy or adding an unmeasured gameplay heuristic.

Validation: backend privacy/robber/API tests, sampled-world/belief/search/reuse
regressions, two upstream friendly-robber tests, all 86 frontend tests, TypeScript
checks and production frontend build passed. Live CNN decision endpoint verified.
