# Shared agent APIs

## Interactive CNN inspection

The user-requested `CNN_INSPECT_SEARCH` and `CNN_INSPECT_POLICY` providers pin
the same XD-CNN D2/W4 seed-23 checkpoint. Select **CNN best measured — search
(500 ms)** or **CNN best measured — policy** in the two-player lobby. Both use
the standard observation boundary and support friendly robber. They are disabled
for four-player games. The search configuration uses the measured four-worker,
c=1.5 baseline without exploration noise.

Selection evidence: 20/50 wins against JSettlers in the historical phase-six
standard-robber search evaluation, the highest completed 50-game CNN tuning
score. This is not confirmation of friendly-robber superiority or a promotion
of the fresh model currently in development. Checkpoint path, checksum and
evidence are pinned in `experiments/catan-ai/web-cnn-inspection.json`.
This explicitly requested web inspection option does not alter training defaults
or the legacy neural-provider selection. Restart the local server after adding
provider classes; saved games remain in `data/games.sqlite3`.

The learned-agent experiment's observation-only encoder and exact two-player
resource ledger are documented in
[the schema-3 contract](../experiments/catan-ai/OBSERVATIONS.md). They accept the
same perspective-specific `web_view` boundary.

Phase 4 adds `STOCHASTIC_REFERENCE`, a diagnostic observation-only stochastic
search provider with a deterministic evaluator and frozen heuristic opponent.
It supports the fixed two-player 15-point rules. See
[the search contract](../experiments/catan-ai/SEARCH.md); this provider contains
no trained model and makes no playing-strength claim.

Phase 5 registers `NEURAL_POLICY` and `NEURAL_SEARCH`, backed by verified D2/W0
PyTorch checkpoints. Both support the fixed two-player rules and share the
existing Python and HTTP decision APIs. The server selects the checkpoint in
`experiments/catan-ai/phase5-selected.json`; clients cannot upload checkpoint
paths. The default is provisional while full architecture training and tuning
are in progress. See [neural training and evaluation](../experiments/catan-ai/NEURAL.md).

`agents.json` is the central provider catalog for Python scripts and the web
server. The browser discovers providers through `GET /api/agents`. Each
provider owns its protocol translation, native state, and decision code.
Requests cannot supply executable commands or Python imports.

## JSettlers

The installed source is [JSettlers 2.7.00](https://github.com/jdmonin/JSettlers2),
revision `7a756baac6ad675f0930605bfb3a8e1bd7a796a5`, under
`integrations/jsettlers/upstream/`. Its original GPLv3 license and source are
retained. Gson 2.8.6 is pinned and verified by SHA-256 in the installer.

The provider instantiates the actual upstream `SOCRobotBrain`, using the
upstream SMART strategy and standard robot parameters. It invokes the brain's
native opening strategy, build planner, negotiator for bank/port trades,
development-card logic, robber strategy, and discard strategy. The previous
local utility scorer and estimator-only policy have been removed.

A persistent Java worker maintains a separate `SOCGame`, players, production
numbers, inventories, ports, and `SOCPlayerTracker` objects for each game/agent
seat. Every public building action is replayed into the native game and its
trackers. The adapter derives coordinates by matching board topology,
including port edges; terrain, dice numbers, and port resources come from the
observation. No fixed terrain or Catanatron node IDs are assumed.

Each native client receives its own hand and public information about other
players. Public resource receipts, costs, trades, and discards update native
resource knowledge; hidden steals convert affected knowledge to UNKNOWN.
Neither future rolls nor the ordered development deck enter the API.

The API layer updates seated providers after successful game creation and
every committed action, including human actions. The UI and Catanatron backend
contain no JSettlers synchronization code. Standalone callers can use the same
sync API. Every decision also catches up from the observation's action history.
A restarted/evicted worker reconstructs the native board from that history and
replans. Saved Catanatron games remain authoritative.

The brain runs synchronously with a capturing native client instead of its
socket/timer loop. Its build requests and subsequent placements become one
Catanatron action. Native bank trades, resource IDs, robber/victim choices,
free roads, and development cards are translated at this boundary and checked
against the supplied legal actions. Rule-constrained robber choices and native
rejection/replanning handle translation differences. Failures surface as
errors; there is no random bot or local scoring fallback.

If a native bank trade requests a resource absent from Catanatron's finite
bank, the adapter sends the upstream `SOCRejectOffer` response back to the
brain. Its own denial limit and planner choose the next action. Synchronization
diagnostics expose cumulative `translation_rejections`; these are retained in
benchmark logs separately from unrecovered provider errors.

Supported games use a 19-hex BASE/TOURNAMENT board with two or four players.
The native winning target follows the game (15 for the default two-player
setup); Catanatron supplies discard counts, including its two-player limit of
9 cards. Player trading is disabled, matching the current UI. Other board
variants are rejected. This integration does not imply optimal play or a
measured strength rating.

## Install and rebuild

Use the project's Conda environment for Python. A JDK (Java 8+) with `java`,
`javac`, and `jar` on PATH is also required; the machine's installed Java 8 is
used here.

```sh
./integrations/jsettlers/install.sh
.conda/bin/python -m pip install -e .
```

The installer preserves the pinned source checkout and downloads the checked
Gson dependency. To rebuild adapter changes without downloading:

```sh
./integrations/jsettlers/build.sh
```

`setup.sh` installs the Python package and rebuilds Java when the downloaded
dependency is present. Restart `./play.sh` after Python/manifest/JAR changes.
No API key, account, additional listening port, or independent game server is
required.

## Python API

```python
from agent_apis import register_agents, choose_action, synchronize, synchronize_game
from catanatron import Game
from catanatron.registry import REGISTRY
from catanatron.serialization import action_from_json, web_view

register_agents()
game = Game(REGISTRY.build_all(["JSETTLERS", "JSETTLERS"]),
            vps_to_win=15, discard_limit=9)
synchronize_game(game)  # initialize all seated stateful providers

view = web_view(game, game.state.current_color())
status = synchronize("JSETTLERS", view)  # update without choosing/playing
chosen = choose_action("JSETTLERS", view)
game.execute(action_from_json(chosen))
synchronize_game(game)  # optional after each action for continuous updates

# Registered Player.decide also uses the same adapter and catches up itself:
winner = game.play()
```

Runnable self-play example: `.conda/bin/python examples/play_agents.py`.

## HTTP API

The launcher mounts these endpoints on the existing loopback server:

- `GET /api/agents`: provider catalog, availability, capabilities, endpoints.
- `POST /api/agents/JSETTLERS/sync`: body
  `{"observation": <web_view>, "color": "RED"}`. `color` is optional and defaults
  to the active seat. Returns native state diagnostics, including engine,
  strategy, state index, board layout, player counts, port flags, and visible
  resource knowledge. It does not advance the authoritative game.
- `POST /api/agents/JSETTLERS/decide`: body
  `{"observation": <web_view>}`; returns
  `{"action": [color, action_type, value]}`. It synchronizes first, then chooses
  a supplied legal action. Submit that action to the normal game API to play it.

Obtain a seat's observation with
`GET /api/games/<id>/states/latest?as=<COLOR>` or `web_view(game, color)`.
A decision observation must contain the active seat's hand. The caller is
responsible for providing an accurate observation and its current legal moves;
the gameplay API validates submitted actions. Requests are limited to 2 MB.

The Java worker serializes requests, uses a 45-second response timeout, and
retains up to 128 game/seat views. Observations include the history needed to
rebuild after process restart, history rewind, or view eviction. Identical
decision requests are idempotent within the worker. A restart can select a
different legal plan because upstream strategies use randomness.

If post-commit synchronization fails, the response has `X-Agent-Sync: failed`
and the server logs the error. The already committed move still succeeds;
the next decision rebuilds from the latest observation. Explicit sync/decide
requests return errors directly.

## Add a provider

Implement `availability()`, `choose(observation)`, and
`Player.decide(game, playable_actions)`. Stateful providers can additionally
implement `synchronize(observation, color=None)`. Add the class import path and
metadata to `agents.json`. The shared API hook handles seated providers without
provider-specific changes to the UI or game engine. Keep credentials out of the
public catalog.

## Verification

```sh
.conda/bin/python -m unittest -v tests.test_agent_apis tests.test_launcher
```

Tests run real Java decisions, check HTTP/Python agreement, verify updates after
human moves, exercise bank trades and building, restart during setup and free
roads, compare native board/ports/resources with Catanatron, and finish two-
and four-player games using legal moves.
