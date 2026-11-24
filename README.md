# Local Catanatron

The official [Catanatron framework](https://github.com/bcollazo/catanatron) is
in `framework/`, based on commit `ecf931181b9a65bb4116a2153fb78c16f1438e00`
(version 3.3.0). The Python game engine, API, legal moves, and persistence remain
unchanged. The React interface uses an illustrated island and table layout
inspired by the supplied Colonist screenshots, with original SVG artwork.

## Repository layout

- `agent_apis/` contains agent interfaces, training, and the native worker.
- `framework/` contains the game engine, server, and browser UI.
- `integrations/` contains external agent integrations; `examples/` contains usage examples.
- `tests/` contains repository regression tests and fixtures.
- `experiments/catan-ai/` records experiment protocols, configurations, and historical results. The current training plan is in `AI_EXPERIMENT_PLAN.md`.
- `rust-catan-testing/` contains native-engine experiments and their evidence. Local build caches and installed dependencies there are ignored by Git.
- `data/` and `runs/` contain local game data and training artifacts; they are ignored by Git and should be preserved.

Historical experiment files retain their original paths because scripts and result provenance reference them. See [current collection workflow](experiments/catan-ai/PARALLEL_CPU_COLLECTION.md) and [visible-robber correction](experiments/catan-ai/VISIBLE_ROBBER_PRIVACY_FIX.md) before starting training.

## Play

```sh
cd /Users/drew/Desktop/Catanotron
./play.sh
```

The launcher uses the project-local **Conda environment** `.conda/` and opens
the starting menu at http://127.0.0.1:8765/. Choose **2 or 4 players** (default: 2),
set their names, select each seat's provider, and start the game.

**Human** and **JSettlers** are available. Phase-4 stochastic reference and
phase-5 neural policy/search providers also appear in the catalog; they support
the fixed two-player rules and their playing-strength evaluation is pending.
Agents are loaded from the shared
`agent_apis/agents.json` registry through `GET /api/agents`. JSettlers runs the
upstream `SOCRobotBrain` with SMART planning and a parallel native board for
each agent seat. The adapter handles state updates, bank/port trades, building,
and development cards. Player-to-player trading is disabled.
See [agent API instructions](agent_apis/README.md) for the Python and HTTP APIs.

Human-only games share one browser with open hands. Against a bot, only the human hand is shown, including while the bot searches. Controls, cards, and the turn banner
follow the active seat. Click the green highlighted intersections and roads
during setup, then click the building preview to confirm. Direct board clicks
also show a confirmation with the resource cost. Selecting a build button first
places immediately on the next valid location; click the button again to cancel. Later, use the bottom actions for building, bank trades,
development cards, rolling, and ending the turn. The robber and discard
panels appear above the hand. Click hand cards to offer or discard them, and
click resources above to request them. A trade-hand click selects a full bank
exchange quantity (4 normally, or 3/2 with ports). Bank confirmation enables only for a
valid exchange, including port rates; player trading is disabled. Year of Plenty
uses this same panel to select two resources. Help and game settings are available on the left.

New games use the BASE map and friendly robber. Two-player games use **15 points
to win** and **discard above 9 cards**. Four-player games use 10 points and
discard above 7 cards. Two-player games use the framework's simulation rules; no
separate official two-player rules variant is added. Click the dice or fast-forward to roll; they roll automatically after three seconds. There is
no overall turn time limit. Existing saved games retain their stored rules.

Keep the launcher running. Ctrl+C stops it. Games persist in
`data/games.sqlite3`. Bookmark a game's URL to resume it after restarting the
launcher. Display names are stored in this browser's local storage.

```sh
./play.sh --port 8766      # another port
./play.sh --no-browser    # print the menu URL
./play.sh --quick-start   # skip the menu; create two human seats
```

Activation is optional; this also works:

```sh
conda activate /Users/drew/Desktop/Catanotron/.conda
python play_catan.py
```

## Setup and rebuild

With Conda installed:

```sh
./setup.sh
```

`environment.yml` specifies Python 3.12 and Node.js 24. The script installs
Python and browser dependencies, then builds the React UI. The launcher
serves the UI and API together on loopback; Docker and PostgreSQL are unnecessary.
To rebuild after frontend edits without reinstalling dependencies:

```sh
PATH="$PWD/.conda/bin:$PATH" CTRON_API_URL='' .conda/bin/npm --prefix framework/ui run build
```

Reload the browser after rebuilding. Backend restarts are only needed after
Python changes.

The new UI lives in `framework/ui/src/game/`, `pages/HomePage.tsx`, and
`pages/GameScreen.tsx`. The central registry is `agent_apis/agents.json`; implementations and shared
Python/HTTP APIs live in `agent_apis/`. The browser discovers agents from the
server, with no separate frontend provider list to edit. JSettlers source and
build tools are in `integrations/jsettlers/`.
The original replay viewer remains at `/replays/<game-id>`. Reference images in
`screenshots/` are ignored by Git.

## Verification

For bounded JSettlers 1v1 self-play data and Catanatron policy/value training
loaders, see [dataset instructions](DATASETS.md). The generated corpus is in
`data/jsettlers-corpora/`.

```sh
.conda/bin/python -m unittest -v tests.test_launcher tests.test_agent_apis
PATH="$PWD/.conda/bin:$PATH" .conda/bin/npm --prefix framework/ui test -- --run
PATH="$PWD/.conda/bin:$PATH" .conda/bin/node framework/ui/node_modules/typescript/bin/tsc --noEmit -p framework/ui/tsconfig.json
```

Checks cover menu selection, names, legal moves and human turn handoff, bank
trades, sequential discards, free roads, replays, error recovery, saved-game
recovery, and browser asset serving.

## Native Rust self-play

The production collector supports native Rust MCTS and ONNX CPU inference with Python-trainable, replay-verified compact journals:

```sh
python -m agent_apis.neural.parallel_cohorts --hours 3 --workers 12 --engine rust
```

Neural games use Rust; JSettlers retains the existing Python/Java path. The pinned W4 trainer, optimizer, whole-game splits and 2,000-game update gate remain unchanged. See [native worker instructions](agent_apis/neural/native_worker/README.md) for standalone Exphormer collection, installation, journal APIs and tests.
