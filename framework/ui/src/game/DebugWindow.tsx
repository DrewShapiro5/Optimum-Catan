import { useEffect, useState } from "react";
import type {
  Color,
  DebugActionEntry,
  DebugSnapshot,
  GameAction,
  GameState,
  MctsOpponentMode,
  ResourceCard,
} from "../utils/api.types";
import { COLOR_HEX, DEV_CARDS, LABELS } from "./model";

const DEV_LABELS: Record<(typeof DEV_CARDS)[number], string> = {
  KNIGHT: "Knight",
  MONOPOLY: "Monopoly",
  YEAR_OF_PLENTY: "Year of Plenty",
  ROAD_BUILDING: "Road Building",
  VICTORY_POINT: "Victory Point",
};

const BOARD_ACTION_TYPES = new Set([
  "BUILD_ROAD",
  "BUILD_SETTLEMENT",
  "BUILD_CITY",
  "MOVE_ROBBER",
]);

export function isBoardAction(action: GameAction): boolean {
  return BOARD_ACTION_TYPES.has(action[1]);
}

export function formatPct(value: number): string {
  const pct = value * 100;
  if (pct >= 10) return `${pct.toFixed(1)}%`;
  if (pct >= 0.1 || pct === 0) return `${pct.toFixed(1)}%`;
  return "<0.1%";
}

export function formatTradeLabel(value: (ResourceCard | null)[]): string {
  const giving = value.slice(0, 4).filter((r): r is ResourceCard => !!r);
  const receiving = value[4];
  if (!giving.length || !receiving) return "Maritime Trade";
  const giveRes = LABELS[giving[0]] ?? giving[0];
  const recvRes = LABELS[receiving] ?? receiving;
  return `${giving.length} ${giveRes} → 1 ${recvRes}`;
}

export function describeTile(
  state: GameState,
  coord: [number, number, number],
): string {
  const placed = state.tiles.find(
    (t) =>
      t.coordinate[0] === coord[0] &&
      t.coordinate[1] === coord[1] &&
      t.coordinate[2] === coord[2],
  );
  if (!placed) return `(${coord.join(",")})`;
  if (placed.tile.type === "DESERT") return "Desert";
  if (placed.tile.type === "RESOURCE_TILE") {
    return `${placed.tile.number} ${LABELS[placed.tile.resource] ?? placed.tile.resource}`;
  }
  return `(${coord.join(",")})`;
}

export function formatDebugAction(
  action: GameAction,
  state: GameState,
): string {
  const [, type, val] = action;
  switch (type) {
    case "BUILD_ROAD":
      return `Road ${(val as [number, number])[0]}–${(val as [number, number])[1]}`;
    case "BUILD_SETTLEMENT":
      return `Settlement #${val}`;
    case "BUILD_CITY":
      return `City #${val}`;
    case "MOVE_ROBBER": {
      const [coord, victim] = val as [[number, number, number], string?];
      const tile = describeTile(state, coord);
      return victim ? `Robber → ${tile} (steal ${victim})` : `Robber → ${tile}`;
    }
    case "MARITIME_TRADE":
      return `Trade ${formatTradeLabel(val as (ResourceCard | null)[])}`;
    case "BUY_DEVELOPMENT_CARD":
      return "Buy Dev Card";
    case "PLAY_KNIGHT_CARD":
      return "Play Knight";
    case "PLAY_ROAD_BUILDING":
      return "Play Road Building";
    case "PLAY_MONOPOLY":
      return `Play Monopoly (${LABELS[val as ResourceCard] ?? val})`;
    case "PLAY_YEAR_OF_PLENTY": {
      const cards = (val as ResourceCard[])
        .map((r) => LABELS[r] ?? r)
        .join(" + ");
      return `Play Year of Plenty (${cards})`;
    }
    case "DISCARD_RESOURCE":
      return `Discard ${LABELS[val as ResourceCard] ?? val}`;
    case "ROLL":
      return "Roll Dice";
    case "END_TURN":
      return "End Turn";
    default:
      return String(type);
  }
}

function actionCategory(action: GameAction): string {
  const t = action[1];
  if (BOARD_ACTION_TYPES.has(t)) return "board";
  if (t === "MARITIME_TRADE") return "trade";
  if (t === "BUY_DEVELOPMENT_CARD" || t.startsWith("PLAY_")) return "dev";
  return "turn";
}

function PolicyRow({
  entry,
  label,
  maxPolicy,
  compact = false,
}: {
  entry: DebugActionEntry;
  label: string;
  maxPolicy: number;
  compact?: boolean;
}) {
  const widthPct =
    maxPolicy > 0 ? Math.max(2, (entry.policy / maxPolicy) * 100) : 0;
  const cat = actionCategory(entry.action);
  return (
    <div
      className={`debug-policy-row ${entry.selected ? "is-selected" : ""} ${compact ? "is-compact" : ""}`}
    >
      <div className="debug-policy-top">
        <span className={`debug-cat-dot cat-${cat}`} aria-hidden="true" />
        <span className="debug-policy-label" title={label}>
          {label}
        </span>
        <strong className="debug-policy-pct">{formatPct(entry.policy)}</strong>
      </div>
      <div className="debug-bar-track">
        <div
          className={`debug-bar-fill cat-${cat}`}
          style={{ width: `${Math.min(100, widthPct)}%` }}
        />
      </div>
      {(entry.visits > 0 || entry.prior !== null) && (
        <div className="debug-policy-meta">
          {entry.visits > 0 && <span>{entry.visits}v</span>}
          {entry.prior !== null && <span>P {formatPct(entry.prior)}</span>}
          {entry.mean_value !== null && (
            <span>Q {(entry.mean_value * 100).toFixed(1)}%</span>
          )}
        </div>
      )}
    </div>
  );
}

export default function DebugWindow({
  open,
  onToggleOpen,
  waitBetweenActions,
  onToggleWait,
  onStepAction,
  onUndoAction,
  onChangeSimulations,
  onChangeMctsOpponent,
  canStep,
  stepping,
  canUndo = false,
  undoing = false,
  snapshot,
  state,
  nameOf,
}: {
  open: boolean;
  onToggleOpen: () => void;
  waitBetweenActions: boolean;
  onToggleWait: (next: boolean) => void;
  onStepAction: () => void;
  onUndoAction?: () => void;
  onChangeSimulations: (sims: number) => void;
  onChangeMctsOpponent?: (mode: MctsOpponentMode) => void;
  canStep: boolean;
  stepping: boolean;
  canUndo?: boolean;
  undoing?: boolean;
  snapshot: DebugSnapshot | null;
  state: GameState;
  nameOf: (color: Color) => string;
}) {
  const [simInput, setSimInput] = useState<string>("");
  const [simDirty, setSimDirty] = useState(false);
  useEffect(() => {
    if (!simDirty && snapshot?.simulations !== undefined && snapshot.simulations !== null) {
      setSimInput(String(snapshot.simulations));
    }
  }, [snapshot?.simulations, simDirty]);
  const applySimulations = () => {
    const parsed = Number(simInput);
    if (!Number.isFinite(parsed)) return;
    const clamped = Math.max(0, Math.min(4096, Math.round(parsed)));
    setSimInput(String(clamped));
    setSimDirty(false);
    onChangeSimulations(clamped);
  };
  const isLive =
    snapshot?.status === "searching" || snapshot?.status === "awaiting_step";
  const actions = snapshot?.actions ?? [];
  const maxPolicy = actions.reduce((m, a) => Math.max(m, a.policy), 0);
  const hasBoardMoves = actions.some((a) => isBoardAction(a.action));
  const simProgress =
    snapshot &&
    snapshot.simulations_target &&
    snapshot.simulations_target > 0
      ? Math.min(
          100,
          (snapshot.simulations_done / snapshot.simulations_target) * 100,
        )
      : null;

  const statusText = (() => {
    if (!snapshot) return "Idle";
    if (snapshot.status === "searching") {
      if (snapshot.simulations_target) {
        return `Searching ${snapshot.simulations_done}/${snapshot.simulations_target}`;
      }
      return `Searching (${snapshot.simulations_done} sims)`;
    }
    if (snapshot.status === "awaiting_step") return "Waiting to step";
    if (snapshot.status === "completed") return "Last move";
    return "Idle";
  })();

  return (
    <div
      className={`debug-dock ${open ? "is-open" : ""}`}
      data-testid="debug-dock"
    >
      <button
        type="button"
        className={`debug-toggle ${isLive ? "is-live" : ""}`}
        onClick={onToggleOpen}
        aria-expanded={open}
        aria-label={open ? "Collapse debug window" : "Expand debug window"}
      >
        <span className="debug-toggle-dot" aria-hidden="true" />
        <span>Debug</span>
        <span className="debug-toggle-caret" aria-hidden="true">
          {open ? "▾" : "▸"}
        </span>
      </button>

      {open && (
        <section
          className="debug-panel cream-panel"
          aria-label="Model debug window"
          data-testid="debug-window"
        >
          <header className="debug-header">
            <div className="debug-title-row">
              <strong>Model Debug</strong>
              <span
                className={`debug-status-pill status-${snapshot?.status ?? "idle"}`}
              >
                {statusText}
              </span>
            </div>
            {snapshot && (
              <div className="debug-actor-row">
                <span
                  className="debug-color-swatch"
                  style={{ background: COLOR_HEX[snapshot.color] }}
                />
                <span className="debug-actor-name">
                  {nameOf(snapshot.color)} · {snapshot.player_label}
                </span>
                {snapshot.root_value !== null && (
                  <span className="debug-value-badge" title="Estimated win probability">
                    Win {(snapshot.root_value * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            )}
            {simProgress !== null && snapshot?.mode === "search" && (
              <div
                className="debug-sim-progress"
                aria-label="MCTS simulation progress"
              >
                <div
                  className="debug-sim-progress-fill"
                  style={{ width: `${simProgress}%` }}
                />
              </div>
            )}
          </header>

          <div className="debug-controls">
            <div className="debug-controls-row">
              <label className="debug-wait-toggle">
                <input
                  type="checkbox"
                  checked={waitBetweenActions}
                  onChange={(e) => onToggleWait(e.target.checked)}
                />
                <span>Wait between actions</span>
              </label>
              <div className="debug-action-btns">
                {onUndoAction && (
                  <button
                    type="button"
                    className="debug-undo-btn"
                    disabled={!canUndo || undoing || stepping}
                    onClick={onUndoAction}
                    title="Move game state back to the previous action"
                  >
                    {undoing ? "Undoing…" : "↶ Undo"}
                  </button>
                )}
                <button
                  type="button"
                  className="debug-step-btn"
                  disabled={!waitBetweenActions || !canStep || stepping || undoing}
                  onClick={onStepAction}
                >
                  {stepping ? "Stepping…" : "Take action →"}
                </button>
              </div>
            </div>
            {snapshot?.can_set_simulations && (
              <div className="debug-sims-row">
                <label htmlFor="debug-sims-input">Search sims</label>
                <input
                  id="debug-sims-input"
                  type="number"
                  aria-label="Search simulations"
                  min={0}
                  max={4096}
                  step={16}
                  value={simInput}
                  onChange={(e) => {
                    setSimInput(e.target.value);
                    setSimDirty(true);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") applySimulations();
                  }}
                />
                <button
                  type="button"
                  className="debug-sims-apply"
                  disabled={simInput.trim() === "" || stepping || undoing}
                  onClick={applySimulations}
                >
                  Set
                </button>
              </div>
            )}
            {snapshot?.can_set_mcts_opponent && onChangeMctsOpponent && (
              <div className="debug-opp-row">
                <label htmlFor="debug-opp-select">MCTS opponent</label>
                <select
                  id="debug-opp-select"
                  aria-label="MCTS opponent mode"
                  value={snapshot.mcts_opponent ?? "self-policy"}
                  disabled={stepping || undoing}
                  onChange={(e) =>
                    onChangeMctsOpponent(e.target.value as MctsOpponentMode)
                  }
                >
                  <option value="self-policy">Self-policy</option>
                  <option value="adversarial-puct">Adversarial PUCT (no-stop)</option>
                  <option value="frozen-mlp">Frozen MLP</option>
                </select>
              </div>
            )}
          </div>

          <div className="debug-scroll">
            <div className="debug-section">
              <div className="debug-section-title">
                <span>Development Card Hand</span>
                <span>
                  {DEV_CARDS.reduce(
                    (sum, k) => sum + (snapshot?.dev_cards?.[k] ?? 0),
                    0,
                  )}{" "}
                  held
                </span>
              </div>
              <div className="debug-dev-grid">
                {DEV_CARDS.map((card) => {
                  const count = snapshot?.dev_cards?.[card] ?? 0;
                  return (
                    <div
                      key={card}
                      className={`debug-dev-chip ${count > 0 ? "has-cards" : ""}`}
                    >
                      <span>{DEV_LABELS[card]}</span>
                      <b>{count}</b>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="debug-section">
              <div className="debug-section-title">
                <span>Policy Distribution</span>
                <span>{actions.length} legal</span>
              </div>
              {hasBoardMoves && (
                <div className="debug-legend" aria-label="Board overlay legend">
                  <span>Board policy:</span>
                  <span className="debug-legend-bar" />
                  <span>0% → max</span>
                </div>
              )}
              {actions.length === 0 ? (
                <p className="debug-empty">
                  Waiting for a model player decision…
                </p>
              ) : (
                <div className="debug-list">
                  {actions.map((entry, idx) => (
                    <PolicyRow
                      key={`act-${idx}-${JSON.stringify(entry.action)}`}
                      entry={entry}
                      label={formatDebugAction(entry.action, state)}
                      maxPolicy={maxPolicy}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
