# Phase 7 early-stop results

Closed early at the user request on November 12, 2026. All completed runs and committed partial games are preserved. Fixed-2,048-simulation games and remaining confirmation are abandoned; the original phase-seven gate is not claimed complete.

| Seed / variant / opponent | Games | Wins | Cutoffs |
| --- | ---: | ---: | ---: |
| 11:policy:JSETTLERS | 50 | 12 | 0 |
| 11:time500-c150:JSETTLERS | 50 | 11 | 0 |
| 23:policy:JSETTLERS | 50 | 5 | 2 |
| 23:time500-c150:JSETTLERS | 50 | 9 | 0 |
| 47:policy:JSETTLERS | 50 | 9 | 0 |
| 47:time500-c150:JSETTLERS | 50 | 10 | 0 |
| 11:time500-c075:JSETTLERS | 50 | 10 | 0 |
| 11:time500-c300:JSETTLERS | 50 | 13 | 0 |
| 11:time100:JSETTLERS | 50 | 15 | 0 |
| 11:heuristic500:JSETTLERS | 50 | 13 | 0 |
| 11:time2000:JSETTLERS | 50 | 13 | 0 |
| 11:sim128:JSETTLERS | 50 | 17 | 0 |
| 11:sim512:JSETTLERS | 10 | 0 | 0 |

Partial active runs: [{"folder": "runs/catan-ai/phase5q-XD-CNN-seed11-simulations-39f43efc05fb-20261112T100621049003Z", "completed_games": 45, "wins": 13, "cutoffs": 0, "errors": 0, "progress": {"completed_games": 45, "games_bytes": 35616, "timings_bytes": 12696477, "games_sha256": "15825eb33e685954b645717e6b02019b8105da8e40c243b040077e15fa8fe9b8", "timings_sha256": "aec1d0bd75270b22d619fa577e767ae457001eea89267e45c691deb5d2f0f7e5", "elapsed_seconds": 24859.399769917014, "configuration_sha256": "39f43efc05fb983dc6c393daa3397176a92dfcaef48842afdf435e3ac8887353"}, "status": "user-stopped; descriptive only"}]

500 ms c=3 won 13/50 versus 12/50 policy; c=0.75 won 10/50 and c=1.5 won 11/50. Heuristic value at 500 ms and neural search at 2 seconds each won 13/50. Fixed 128 simulations won 17/50, but is not an equal-time result. The 100 ms result fails its latency gate.

Paired comparisons are descriptive after user-requested stopping and tuning selection: {"time500-c075": {"difference": -0.04, "ci95": [-0.16, 0.06], "descriptive_only": true}, "time500-c150": {"difference": -0.02, "ci95": [-0.14, 0.1], "descriptive_only": true}, "time500-c300": {"difference": 0.02, "ci95": [-0.12, 0.14], "descriptive_only": true}, "heuristic500": {"difference": 0.02, "ci95": [-0.06, 0.1], "descriptive_only": true}, "time2000": {"difference": 0.02, "ci95": [-0.14, 0.2], "descriptive_only": true}, "sim128": {"difference": 0.1, "ci95": [-0.06, 0.26], "descriptive_only": true}}

Continue with XD-CNN D2/W0, c=1.5/C0 as provisional unchanged settings. No statistical search superiority or final-set result is claimed. Before additional strategic complexity, perform a bounded phase-eight inference-cache correctness/throughput and value/opponent diagnostic. No long match queue is automatically launched.
