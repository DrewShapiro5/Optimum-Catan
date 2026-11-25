import { useCallback, useContext, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { store } from "../store";
import ACTIONS from "../actions";
import {
  getDebugState,
  getState,
  postAction,
  prepareDebugAction,
  setDebugSearchSettings,
  setDebugSimulations,
  setPerspective,
  stepDebugAction,
  undoDebugAction,
  type StateIndex,
} from "../utils/apiClient";
import type {
  DebugSnapshot,
  GameAction,
  GameState,
  MctsOpponentMode,
  ResourceCard,
} from "../utils/api.types";
import Icon, { type IconName } from "../game/Icons";
import Card from "../game/Cards";
import ResourcePanel, { type ResourceMode } from "../game/ResourcePanel";
import { useAutoRoll } from "../game/useAutoRoll";
import IslandBoard from "../game/IslandBoard";
import DebugWindow, { isBoardAction } from "../game/DebugWindow";
import { Bank, Dice, GameLog, PlayerPanel } from "../game/TablePanels";
import {
  COLOR_HEX,
  DEV_CARDS,
  LABELS,
  RESOURCES,
  playerNames,
  stat,
  turnPrompt,
} from "../game/model";
import "../game/colonist.scss";

type Modal = "dev" | "help" | "settings" | "victim" | null;
export default function GameScreen({ replayMode }: { replayMode: boolean }) {
  const { gameId, stateIndex } = useParams();
  const { state, dispatch } = useContext(store),
    game = state.gameState;
  const [loaded, setLoaded] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [mode, setMode] = useState<string | null>(null),
    [modal, setModal] = useState<Modal>(null),
    [zoom, setZoom] = useState(1);
  const [victims, setVictims] = useState<GameAction[]>([]);
  const [resourceMode, setResourceMode] = useState<ResourceMode | null>(null);
  const [give, setGive] = useState<ResourceCard[]>([]);
  const [pending, setPending] = useState<{
    action: GameAction;
    state: number;
  } | null>(null);
  const [debugOpen, setDebugOpen] = useState(false);
  const [waitBetweenActions, setWaitBetweenActions] = useState(false);
  const [debugSnapshot, setDebugSnapshot] = useState<DebugSnapshot | null>(
    null,
  );
  const [debugPreparing, setDebugPreparing] = useState(false);
  const [debugStepping, setDebugStepping] = useState(false);
  const [debugUndoing, setDebugUndoing] = useState(false);
  const [prepareVersion, setPrepareVersion] = useState(0);
  const preparedStateRef = useRef<string | null>(null);
  const preparingRef = useRef(false);
  const actionLock = useRef(false);
  const modalRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!modal || !modalRef.current) return;
    const previous = document.activeElement as HTMLElement | null;
    const panel = modalRef.current;
    const selectable = () =>
      Array.from(
        panel.querySelectorAll<HTMLElement>(
          'button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), [tabindex="0"]',
        ),
      );
    (selectable()[0] ?? panel).focus();
    const trap = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const items = selectable();
      const first = items[0],
        last = items[items.length - 1];
      if (!first) {
        e.preventDefault();
        panel.focus();
      } else if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    panel.addEventListener("keydown", trap);
    return () => {
      panel.removeEventListener("keydown", trap);
      previous?.focus();
    };
  }, [modal]);
  const names = playerNames(gameId ?? "");
  const nameOf = (color: GameState["current_color"]) =>
    names[color] ?? `${color[0]}${color.slice(1).toLowerCase()}`;
  const publish = useCallback(
    (data: GameState) => dispatch({ type: ACTIONS.SET_GAME_STATE, data }),
    [dispatch],
  );
  useEffect(() => {
    let cancelled = false;
    setLoaded(false);
    setError("");
    setMode(null);
    setModal(null);
    setResourceMode(null);
    setPending(null);
    setDebugSnapshot(null);
    setDebugPreparing(false);
    setDebugStepping(false);
    setDebugUndoing(false);
    preparedStateRef.current = null;
    preparingRef.current = false;
    setPerspective(null);
    if (gameId)
      getState(gameId, stateIndex as StateIndex)
        .then((data) => {
          if (!cancelled) {
            publish(data);
            setLoaded(true);
          }
        })
        .catch(() => {
          if (!cancelled)
            setError(
              "We couldn't load this game. Check the server and try again.",
            );
        });
    return () => {
      cancelled = true;
    };
  }, [gameId, stateIndex, publish]);
  const perform = useCallback(
    async (action?: GameAction | GameAction[]) => {
      if (!gameId || actionLock.current || replayMode) return;
      actionLock.current = true;
      setBusy(true);
      setError("");
      try {
        const batch =
          action && Array.isArray(action[0])
            ? (action as GameAction[])
            : [action as GameAction | undefined];
        for (const a of batch) publish(await postAction(gameId, a));
        setMode(null);
        setPending(null);
        setResourceMode(null);
        setModal(null);
        setGive([]);
      } catch {
        setError(
          "That action couldn't be completed. The latest game state has been requested; please try again.",
        );
        setMode(null);
        setPending(null);
        setResourceMode(null);
        setModal(null);
        setGive([]);
        try {
          publish(await getState(gameId));
        } catch {
          /* Keep the last displayed state. */
        }
      } finally {
        actionLock.current = false;
        setBusy(false);
      }
    },
    [gameId, publish, replayMode],
  );
  const handleStepAction = useCallback(async () => {
    if (!gameId || actionLock.current || debugStepping || replayMode) return;
    actionLock.current = true;
    setDebugStepping(true);
    setError("");
    try {
      const nextState = await stepDebugAction(gameId);
      if (nextState) publish(nextState);
      preparedStateRef.current = null;
    } catch {
      setError(
        "That action couldn't be completed. The latest game state has been requested; please try again.",
      );
      try {
        publish(await getState(gameId));
      } catch {
        /* Keep the last displayed state. */
      }
    } finally {
      actionLock.current = false;
      setDebugStepping(false);
    }
  }, [gameId, debugStepping, replayMode, publish]);
  const handleUndoAction = useCallback(async () => {
    if (!gameId || actionLock.current || debugUndoing || replayMode) return;
    actionLock.current = true;
    setDebugUndoing(true);
    setError("");
    try {
      const prevState = await undoDebugAction(gameId);
      if (prevState) {
        if (prevState.bot_colors.includes(prevState.current_color)) {
          setWaitBetweenActions(true);
        }
        publish(prevState);
        setMode(null);
        setPending(null);
        setResourceMode(null);
        setModal(null);
        setGive([]);
      }
      preparedStateRef.current = null;
      setDebugSnapshot(null);
      setPrepareVersion((v) => v + 1);
    } catch {
      setError("Could not undo the previous action.");
      try {
        publish(await getState(gameId));
      } catch {
        /* Keep the last displayed state. */
      }
    } finally {
      actionLock.current = false;
      setDebugUndoing(false);
    }
  }, [gameId, debugUndoing, replayMode, publish]);
  const handleChangeSimulations = useCallback(
    async (sims: number) => {
      if (!gameId || replayMode) return;
      try {
        const snap = await setDebugSimulations(gameId, sims);
        if (snap) setDebugSnapshot(snap);
        preparedStateRef.current = null;
        setPrepareVersion((v) => v + 1);
      } catch {
        setError("Could not update search simulations.");
      }
    },
    [gameId, replayMode],
  );
  const handleChangeMctsOpponent = useCallback(
    async (mctsOpponent: MctsOpponentMode) => {
      if (!gameId || replayMode) return;
      try {
        const snap = await setDebugSearchSettings(gameId, {
          mcts_opponent: mctsOpponent,
        });
        if (snap) setDebugSnapshot(snap);
        preparedStateRef.current = null;
        setPrepareVersion((v) => v + 1);
      } catch {
        setError("Could not update MCTS opponent mode.");
      }
    },
    [gameId, replayMode],
  );
  const handleChangeVpMarginAlpha = useCallback(
    async (alpha: number) => {
      if (!gameId || replayMode) return;
      try {
        const snap = await setDebugSearchSettings(gameId, {
          vp_margin_alpha: alpha,
        });
        if (snap) setDebugSnapshot(snap);
        preparedStateRef.current = null;
        setPrepareVersion((v) => v + 1);
      } catch {
        setError("Could not update VP-margin alpha.");
      }
    },
    [gameId, replayMode],
  );
  useEffect(() => {
    if (
      !loaded ||
      !game ||
      game.game.id !== gameId ||
      replayMode ||
      game.winning_color ||
      !game.bot_colors.includes(game.current_color) ||
      busy ||
      debugStepping ||
      debugUndoing ||
      error
    )
      return;
    if (!waitBetweenActions) {
      if (preparingRef.current) return;
      void perform();
      return;
    }
    const stateKey = `${gameId}:${game.state_index}:${prepareVersion}`;
    if (
      preparedStateRef.current === stateKey ||
      preparingRef.current ||
      debugStepping ||
      debugUndoing
    )
      return;
    preparedStateRef.current = stateKey;
    preparingRef.current = true;
    setDebugPreparing(true);
    let cancelled = false;
    prepareDebugAction(gameId)
      .then((snap) => {
        if (!cancelled && snap) setDebugSnapshot(snap);
      })
      .catch(() => {
        if (!cancelled) preparedStateRef.current = null;
      })
      .finally(() => {
        preparingRef.current = false;
        if (!cancelled) setDebugPreparing(false);
      });
    return () => {
      cancelled = true;
    };
  }, [
    loaded,
    game,
    gameId,
    replayMode,
    busy,
    error,
    waitBetweenActions,
    debugStepping,
    debugUndoing,
    prepareVersion,
    perform,
  ]);
  useEffect(() => {
    if (!debugOpen || !loaded || !game || game.game.id !== gameId || replayMode)
      return;
    let cancelled = false;
    const poll = () => {
      getDebugState(gameId)
        .then((snap) => {
          if (!cancelled && snap) setDebugSnapshot(snap);
        })
        .catch(() => {
          /* Ignore transient debug telemetry errors. */
        });
    };
    poll();
    const isBotSearching =
      (busy || debugPreparing) && game.bot_colors.includes(game.current_color);
    if (!isBotSearching) {
      return () => {
        cancelled = true;
      };
    }
    const timer = window.setInterval(poll, 120);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [debugOpen, loaded, game, gameId, replayMode, busy, debugPreparing]);
  useEffect(() => {
    setPending(null);
  }, [gameId, game?.state_index, mode]);
  useEffect(() => {
    if (
      loaded &&
      game &&
      game.game.id === gameId &&
      game.current_prompt === "DISCARD" &&
      !game.bot_colors.includes(game.current_color) &&
      !busy &&
      !replayMode
    ) {
      setResourceMode("discard");
      setGive([]);
    }
  }, [
    loaded,
    gameId,
    game?.current_color,
    game?.current_discard_count,
    game?.current_prompt,
    busy,
    replayMode,
  ]);
  const rollAction =
    loaded &&
    game &&
    game.game.id === gameId &&
    !game.winning_color &&
    !replayMode &&
    !game.is_initial_build_phase &&
    !game.bot_colors.includes(game.current_color)
      ? game.current_playable_actions.find((a) => a[1] === "ROLL")
      : undefined;
  const rollTurn =
    rollAction && game
      ? `${gameId}:${game.current_color}:${game.num_turns}:${game.current_turn_index}`
      : null;
  const rollSeconds = useAutoRoll(
    rollTurn,
    busy || !!error || !!modal || !!resourceMode,
    () => {
      if (rollAction) void perform(rollAction);
    },
  );
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !actionLock.current) {
        setModal(null);
        setMode(null);
        setPending(null);
        setResourceMode(null);
        setGive([]);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  if (!loaded || !game)
    return (
      <div className="settlers-app game-loading">
        <div className="cream-panel">
          <Icon name="settlement" />
          <h1>{error ? "Game unavailable" : "Loading game…"}</h1>
          {error && (
            <>
              <p role="alert">{error}</p>
              <button
                className="start-game"
                onClick={() => window.location.reload()}
              >
                Try again
              </button>
              <Link to="/">Back to menu</Link>
            </>
          )}
        </div>
      </div>
    );
  const color = game.current_color,
    handColor = game.viewer_color ?? color,
    canPlay =
      !busy &&
      !replayMode &&
      !game.winning_color &&
      !game.bot_colors.includes(color);
  const legal = game.current_playable_actions;
  const find = (type: string) => legal.find((a) => a[1] === type);
  const has = (type: string) => !!find(type);
  const lastRoll = [...game.action_records]
    .reverse()
    .find((r) => r[0][1] === "ROLL");
  const rolls = lastRoll?.[1] as number[] | undefined;
  const prompt = mode
    ? `Choose a location for your ${mode.replace("BUILD_", "").toLowerCase()}`
    : turnPrompt(game);
  const isDiscard = game.current_prompt === "DISCARD",
    isRobber = game.current_prompt === "MOVE_ROBBER";
  const roadPrompt =
    game.current_prompt === "BUILD_ROAD" ||
    game.current_prompt === "BUILD_INITIAL_ROAD";
  const mainAction = find("ROLL") ?? find("END_TURN");
  const devActions = legal.filter((a) => a[1].startsWith("PLAY_"));
  const availableDevs = has("BUY_DEVELOPMENT_CARD");
  const humanName = game.winning_color
    ? nameOf(game.winning_color)
    : nameOf(color);
  const actionButton = (
    icon: IconName,
    label: string,
    disabled: boolean,
    click: () => void,
    count?: number,
    selected = false,
    cost?: string,
  ) => (
    <button
      className={`action-tile ${selected ? "selected" : ""}`}
      disabled={disabled || !canPlay}
      onClick={click}
      aria-label={label}
      title={cost ? `${label} · ${cost}` : label}
    >
      <Icon name={icon} style={{ color: COLOR_HEX[color] }} />
      {count !== undefined && <span className="piece-count">{count}</span>}
      <span>{label}</span>
    </button>
  );
  const close = () => {
    if (!busy) {
      setModal(null);
      setGive([]);
    }
  };
  const openResources = (next: ResourceMode) => {
    setResourceMode(next);
    setGive([]);
    setModal(null);
    setMode(null);
    setPending(null);
  };
  const bankRate = (r: ResourceCard) =>
    Math.min(
      ...legal
        .filter((a) => a[1] === "MARITIME_TRADE")
        .filter((a) => a[2][0] === r)
        .map((a) => a[2].slice(0, 4).filter(Boolean).length),
    );
  const selectFromHand = (r: ResourceCard) => {
    if (
      busy ||
      !canPlay ||
      (resourceMode !== "trade" && resourceMode !== "discard")
    )
      return;
    if (give.filter((c) => c === r).length >= stat(game, color, `${r}_IN_HAND`))
      return;
    if (resourceMode === "discard" && give.length >= game.current_discard_count)
      return;
    const quantity = resourceMode === "trade" ? bankRate(r) : 1;
    if (
      !Number.isFinite(quantity) ||
      stat(game, color, `${r}_IN_HAND`) - give.filter((c) => c === r).length <
        quantity
    )
      return;
    setGive([...give, ...Array<ResourceCard>(quantity).fill(r)]);
  };
  return (
    <div className="settlers-app game-table">
      <header className="table-header">
        <Link to="/" className="brand">
          <span className="brand-mark">
            <Icon name="settlement" />
          </span>
          CATANATRON
        </Link>
        <span className="table-label">
          CLASSIC ISLAND <i /> {game.colors.length} PLAYERS
        </span>
        <Link className="new-table-link" to="/">
          New game ↗
        </Link>
      </header>
      <section className="board-area">
        <nav className="board-tools" aria-label="Board tools">
          <button
            onClick={() => setModal("settings")}
            title="Game settings"
            aria-label="Game settings"
          >
            <Icon name="settings" />
          </button>
          <button
            onClick={() => setModal("help")}
            title="How to play"
            aria-label="How to play"
          >
            <Icon name="help" />
          </button>
          <button
            onClick={() => setZoom((z) => (z === 1 ? 1.2 : 1))}
            title="Toggle board zoom"
            aria-label="Toggle board zoom"
          >
            <Icon name="expand" />
          </button>
          <Link to="/" title="Back to menu" aria-label="Back to menu">
            <Icon name="back" />
          </Link>
        </nav>
        {!replayMode && (
          <DebugWindow
            open={debugOpen}
            onToggleOpen={() => setDebugOpen((o) => !o)}
            waitBetweenActions={waitBetweenActions}
            onToggleWait={(next) => {
              setWaitBetweenActions(next);
              if (
                !next &&
                debugSnapshot?.status === "awaiting_step" &&
                !debugStepping &&
                !busy
              ) {
                void handleStepAction();
              }
            }}
            onStepAction={() => void handleStepAction()}
            onUndoAction={() => void handleUndoAction()}
            onChangeSimulations={(sims) => void handleChangeSimulations(sims)}
            onChangeMctsOpponent={(mode) =>
              void handleChangeMctsOpponent(mode)
            }
            onChangeVpMarginAlpha={(alpha) =>
              void handleChangeVpMarginAlpha(alpha)
            }
            canStep={
              waitBetweenActions &&
              game.bot_colors.includes(game.current_color) &&
              !game.winning_color &&
              !debugPreparing &&
              !busy &&
              !debugStepping &&
              !debugUndoing
            }
            stepping={debugStepping}
            canUndo={
              game.state_index > 0 &&
              !busy &&
              !debugStepping &&
              !debugUndoing
            }
            undoing={debugUndoing}
            snapshot={debugSnapshot}
            state={game}
            nameOf={nameOf}
          />
        )}
        <div className="island-wrap">
          <IslandBoard
            state={game}
            debugPolicyActions={
              debugOpen &&
              debugSnapshot &&
              debugSnapshot.state_index === game.state_index &&
              (debugSnapshot.status === "searching" ||
                debugSnapshot.status === "awaiting_step")
                ? debugSnapshot.actions.filter((e) => isBoardAction(e.action))
                : null
            }
            mode={mode}
            disabled={!canPlay}
            zoom={zoom}
            pendingAction={
              pending?.state === game.state_index ? pending.action : null
            }
            onConfirm={() => {
              if (
                pending?.state === game.state_index &&
                legal.some(
                  (a) => JSON.stringify(a) === JSON.stringify(pending.action),
                )
              )
                void perform(pending.action);
            }}
            onCancel={() => setPending(null)}
            onAction={(a) => {
              setResourceMode(null);
              if (!game.is_initial_build_phase && mode === a[1])
                void perform(a);
              else setPending({ action: a, state: game.state_index });
            }}
            onRobber={(actions) => {
              if (actions.length === 1) void perform(actions[0]);
              else {
                setVictims(actions);
                setModal("victim");
              }
            }}
          />
        </div>
        <div className="board-caption">
          <span>
            {game.is_initial_build_phase ? "SETUP" : "GAME IN PROGRESS"}
          </span>
          <span>{game.game.vps_to_win} victory points to win</span>
        </div>
        <div className="table-dice">
          <button
            className="dice-control"
            aria-label="Roll dice"
            disabled={!canPlay || !rollAction}
            onClick={() => {
              if (rollAction) void perform(rollAction);
            }}
          >
            <Dice values={rolls} />
          </button>
          {rollAction && rollSeconds !== null ? (
            <span className="roll-countdown">Auto-roll in {rollSeconds}s</span>
          ) : (
            rolls && <span>{rolls[0] + rolls[1]}</span>
          )}
        </div>
      </section>
      <aside className="table-sidebar">
        <GameLog state={game} names={names} />
        <Bank state={game} />
        <div className="local-table-note">
          <i /> Local table <span>{game.viewer_color ? "Private hand" : "Shared screen · open hands"}</span>
        </div>
        <section
          className={`player-list ${game.colors.length === 4 ? "four-seats" : ""}`}
          aria-label="Players"
        >
          {game.colors.map((c) => (
            <PlayerPanel state={game} color={c} name={nameOf(c)} key={c} />
          ))}
        </section>
      </aside>
      <footer className="table-bottom">
        <div
          className="turn-banner cream-panel"
          role="status"
          aria-label="Current turn"
        >
          <span
            className="avatar"
            style={
              {
                "--player-color": COLOR_HEX[game.winning_color ?? color],
              } as React.CSSProperties
            }
          >
            <Icon name="person" />
          </span>
          <div>
            <small>
              {replayMode
                ? `REPLAY · STATE ${game.state_index}`
                : busy
                  ? "UPDATING THE TABLE"
                  : `${humanName.toUpperCase()}${game.winning_color ? "" : "'S TURN"}`}
            </small>
            <strong>{game.winning_color ? `${humanName} wins` : prompt}</strong>
          </div>
          <span className="turn-counter">
            {game.is_initial_build_phase
              ? "SETUP"
              : `TURN ${game.num_turns + 1}`}
          </span>
          {mode && (
            <button
              className="cancel-mode"
              onClick={() => setMode(null)}
              aria-label="Cancel building"
            >
              <Icon name="close" />
            </button>
          )}
        </div>
        <section
          className={`hand-panel cream-panel ${isDiscard ? "discard-warning" : ""} ${resourceMode ? "resource-open" : ""}`}
          aria-label="Current player's hand"
        >
          <div className="panel-title">
            <span>{nameOf(handColor)}’s hand</span>
            <span>
              {RESOURCES.reduce(
                (n, r) => n + stat(game, handColor, `${r}_IN_HAND`),
                0,
              )}{" "}
              resource cards
            </span>
          </div>
          <div className="hand-cards">
            {RESOURCES.filter((r) => stat(game, handColor, `${r}_IN_HAND`) > 0).map(
              (r) => (
                <Card
                  key={r}
                  resource={r}
                  count={
                    stat(game, handColor, `${r}_IN_HAND`) -
                    (resourceMode ? give.filter((c) => c === r).length : 0)
                  }
                  onClick={
                    resourceMode === "trade" || resourceMode === "discard"
                      ? () => selectFromHand(r)
                      : undefined
                  }
                  disabled={
                    busy ||
                    !canPlay ||
                    give.filter((c) => c === r).length >=
                      stat(game, handColor, `${r}_IN_HAND`) ||
                    (resourceMode === "discard" &&
                      give.length >= game.current_discard_count)
                  }
                />
              ),
            )}
            {DEV_CARDS.some((r) => stat(game, handColor, `${r}_IN_HAND`) > 0) && (
              <>
                <span className="hand-divider" />
                {DEV_CARDS.filter(
                  (r) => stat(game, handColor, `${r}_IN_HAND`) > 0,
                ).map((r) => (
                  <button
                    className="dev-hand-card"
                    key={r}
                    onClick={() => setModal("dev")}
                    title={r.replaceAll("_", " ")}
                    disabled={!canPlay}
                  >
                    <Card
                      resource="dev"
                      count={stat(game, handColor, `${r}_IN_HAND`)}
                    />
                    <small>{r.replaceAll("_", " ")}</small>
                  </button>
                ))}
              </>
            )}
            {RESOURCES.every((r) => !stat(game, handColor, `${r}_IN_HAND`)) &&
              !DEV_CARDS.some((r) => stat(game, handColor, `${r}_IN_HAND`) > 0) && (
                <span className="empty-hand">
                  <Icon name="WOOD" />
                  Your resources will appear here.
                </span>
              )}
          </div>
          {resourceMode && (
            <ResourcePanel
              key={`${resourceMode}:${game.current_color}:${game.current_discard_count}`}
              state={game}
              mode={resourceMode}
              give={give}
              setGive={setGive}
              busy={busy || !canPlay}
              onSubmit={(actions) => void perform(actions)}
              onClose={() => {
                setResourceMode(null);
                setGive([]);
              }}
            />
          )}
        </section>
        <div className="table-actions" aria-label="Game actions">
          {actionButton(
            "trade",
            "Trade",
            game.is_initial_build_phase ||
              game.current_prompt !== "PLAY_TURN" ||
              !!game.is_road_building,
            () => openResources("trade"),
            undefined,
            false,
            "Trade resources with the bank",
          )}
          {actionButton(
            "dev",
            "Develop",
            !availableDevs,
            () => {
              const purchase = find("BUY_DEVELOPMENT_CARD");
              if (purchase) void perform(purchase);
            },
            undefined,
            false,
            "Buy: wool + grain + ore",
          )}
          {actionButton(
            "road",
            "Road",
            !has("BUILD_ROAD") || game.is_initial_build_phase || roadPrompt,
            () => setMode(mode === "BUILD_ROAD" ? null : "BUILD_ROAD"),
            stat(game, color, "ROADS_AVAILABLE"),
            mode === "BUILD_ROAD",
            "lumber + brick",
          )}
          {actionButton(
            "settlement",
            "Settle",
            !has("BUILD_SETTLEMENT") || game.is_initial_build_phase,
            () =>
              setMode(mode === "BUILD_SETTLEMENT" ? null : "BUILD_SETTLEMENT"),
            stat(game, color, "SETTLEMENTS_AVAILABLE"),
            mode === "BUILD_SETTLEMENT",
            "lumber + brick + wool + grain",
          )}
          {actionButton(
            "city",
            "City",
            !has("BUILD_CITY"),
            () => setMode(mode === "BUILD_CITY" ? null : "BUILD_CITY"),
            stat(game, color, "CITIES_AVAILABLE"),
            mode === "BUILD_CITY",
            "2 grain + 3 ore",
          )}
          {actionButton(
            isRobber ? "robber" : isDiscard ? "check" : "next",
            isDiscard
              ? "Discard"
              : isRobber
                ? "Move robber"
                : has("ROLL")
                  ? "Roll dice (fast forward)"
                  : "End turn",
            (!isDiscard && !mainAction) || isRobber,
            () => {
              if (isDiscard) openResources("discard");
              else if (mainAction) void perform(mainAction);
            },
          )}
        </div>
      </footer>
      {error && (
        <div className="game-error" role="alert">
          {error}
          <button onClick={() => setError("")} aria-label="Dismiss error">
            ×
          </button>
        </div>
      )}
      {modal && (
        <div className="game-modal-backdrop" onClick={close}>
          <section
            className={`game-modal cream-panel modal-${modal}`}
            ref={modalRef}
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-label={
              modal === "dev"
                ? "Development cards"
                : modal === "victim"
                  ? "Choose a player to rob"
                  : modal === "help"
                    ? "How to play"
                    : "Game settings"
            }
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              onClick={close}
              disabled={busy}
              aria-label="Close dialog"
            >
              <Icon name="close" />
            </button>
            {modal === "dev" && (
              <>
                <h2>Development cards</h2>
                <p>
                  Cards can be played on a later turn. Victory point cards score
                  automatically.
                </p>
                <button
                  className="buy-development"
                  disabled={!canPlay || !has("BUY_DEVELOPMENT_CARD")}
                  onClick={() => {
                    const a = find("BUY_DEVELOPMENT_CARD");
                    if (a) void perform(a);
                  }}
                >
                  <Icon name="dev" />
                  <span>
                    Buy a development card
                    <small>1 wool + 1 grain + 1 ore</small>
                  </span>
                </button>
                <h3>Play a card</h3>
                <div className="development-options">
                  {has("PLAY_YEAR_OF_PLENTY") && (
                    <button
                      disabled={!canPlay}
                      onClick={() => openResources("plenty")}
                    >
                      <Icon name="dev" />
                      <span>
                        Year of Plenty<small>Select two resources</small>
                      </span>
                    </button>
                  )}
                  {devActions
                    .filter((a) => a[1] !== "PLAY_YEAR_OF_PLENTY")
                    .map((a, i) => (
                      <button
                        disabled={!canPlay}
                        key={i}
                        onClick={() => void perform(a)}
                      >
                        <Icon
                          name={
                            a[1] === "PLAY_KNIGHT_CARD"
                              ? "army"
                              : a[1] === "PLAY_ROAD_BUILDING"
                                ? "road"
                                : "dev"
                          }
                        />
                        <span>
                          {a[1]
                            .replace("PLAY_", "")
                            .replaceAll("_", " ")
                            .toLowerCase()}
                          {a[1] === "PLAY_MONOPOLY" && (
                            <small>{LABELS[a[2]]}</small>
                          )}
                        </span>
                      </button>
                    ))}
                </div>
                {devActions.length === 0 && (
                  <p className="muted">
                    No development cards can be played right now.
                  </p>
                )}
              </>
            )}
            {modal === "victim" && (
              <>
                <h2>Who will you rob?</h2>
                <p>Choose a player adjacent to this hex.</p>
                <div className="victim-options">
                  {victims.map((a, i) => (
                    <button
                      key={i}
                      disabled={busy}
                      onClick={() => void perform(a)}
                    >
                      <Icon name="person" />
                      {a[1] === "MOVE_ROBBER" && a[2][1]
                        ? nameOf(a[2][1] as GameState["current_color"])
                        : "No player"}
                    </button>
                  ))}
                </div>
              </>
            )}
            {modal === "help" && (
              <>
                <h2>How to play</h2>
                <p>
                  Reach {game.game.vps_to_win} victory points first. Both humans
                  share this screen, and controls follow the active seat.
                </p>
                <ol className="rules-list">
                  <li>
                    <b>Setup.</b> Click a highlighted intersection for a
                    settlement, then confirm the preview. Choose and confirm a
                    road next. Each player places two of each.
                  </li>
                  <li>
                    <b>Resources.</b> A hex produces resources when its number
                    is rolled. Settlements collect one, cities collect two.
                  </li>
                  <li>
                    <b>Build & trade.</b> Use the action buttons to build roads,
                    settlements, and cities, or trade with the bank. Hover over
                    a button to see its cost.
                  </li>
                  <li>
                    <b>Robber.</b> On a seven, hands above{" "}
                    {game.game.discard_limit} cards discard half. Choose a hex
                    to move the robber and a player to steal from.
                  </li>
                  <li>
                    <b>Scoring.</b> Settlements score 1, cities score 2. Longest
                    road, largest army, and development cards can add points.
                  </li>
                </ol>
              </>
            )}
            {modal === "settings" && (
              <>
                <h2>Game settings</h2>
                <dl className="settings-list">
                  <div>
                    <dt>Players</dt>
                    <dd>{game.colors.length}</dd>
                  </div>
                  <div>
                    <dt>Victory points to win</dt>
                    <dd>{game.game.vps_to_win}</dd>
                  </div>
                  <div>
                    <dt>Discard above</dt>
                    <dd>{game.game.discard_limit} cards</dd>
                  </div>
                  <div>
                    <dt>Friendly robber</dt>
                    <dd>{game.game.friendly_robber ? "On" : "Off"}</dd>
                  </div>
                  <div>
                    <dt>Visibility</dt>
                    <dd>{game.viewer_color ? "Private hand" : "Shared screen · open hands"}</dd>
                  </div>
                </dl>
                <p>
                  Games are saved automatically. Bookmark this game's URL to
                  return to it.
                </p>
                <Link className="start-game" to="/">
                  New game →
                </Link>
              </>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
