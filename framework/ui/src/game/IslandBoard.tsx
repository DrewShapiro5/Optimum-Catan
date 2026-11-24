import { useState } from "react";
import type {
  DebugActionEntry,
  GameAction,
  GameState,
  ResourceCard,
  TileCoordinate,
} from "../utils/api.types";
import { tilePixelVector, getNodeDelta, SQRT3 } from "../utils/coordinates";
import { Artwork } from "./Icons";
import { COLOR_HEX, BUILD_COST } from "./model";
import Icon from "./Icons";
import { describeTile, formatPct } from "./DebugWindow";

const SIZE = 62;
const hex = (r: number) =>
  Array.from(
    { length: 6 },
    (_, i) =>
      `${Math.cos(((i * 60 - 90) * Math.PI) / 180) * r},${Math.sin(((i * 60 - 90) * Math.PI) / 180) * r}`,
  ).join(" ");
const TERRAIN: Record<ResourceCard | "DESERT", [string, string]> = {
  WOOD: ["#54b342", "#238d40"],
  BRICK: ["#f49a52", "#d56a42"],
  SHEEP: ["#c2db42", "#90b526"],
  WHEAT: ["#ffdc68", "#eab63c"],
  ORE: ["#c8d0c4", "#8faaa7"],
  DESERT: ["#e4d697", "#bcb46e"],
};

type HoveredPolicy = {
  x: number;
  y: number;
  title: string;
  policy: number;
  entries: DebugActionEntry[];
};

function policyIntensity(policy: number, maxPolicy: number): number {
  if (maxPolicy <= 0 || policy <= 0) return 0;
  const rel = Math.max(0, Math.min(1, policy / maxPolicy));
  return Math.pow(rel, 0.55);
}

/** Map normalized policy intensity [0, 1] to vivid heat color (cool blue -> amber -> magenta-red). */
function policyColor(t: number, alphaMin = 0.42, alphaMax = 0.96): string {
  const clamped = Math.max(0, Math.min(1, t));
  let r: number, g: number, b: number;
  if (clamped < 0.5) {
    const u = clamped * 2;
    r = Math.round(92 + (255 - 92) * u);
    g = Math.round(196 + (196 - 196) * u);
    b = Math.round(255 + (38 - 255) * u);
  } else {
    const u = (clamped - 0.5) * 2;
    r = Math.round(255 + (255 - 255) * u);
    g = Math.round(196 + (40 - 196) * u);
    b = Math.round(38 + (92 - 38) * u);
  }
  const a = alphaMin + (alphaMax - alphaMin) * clamped;
  return `rgba(${r}, ${g}, ${b}, ${a.toFixed(3)})`;
}

export default function IslandBoard({
  state,
  mode,
  onAction,
  onRobber,
  disabled,
  zoom = 1,
  pendingAction,
  onConfirm,
  onCancel,
  debugPolicyActions = null,
}: {
  state: GameState;
  mode: string | null;
  onAction: (a: GameAction) => void;
  onRobber: (a: GameAction[]) => void;
  disabled: boolean;
  zoom?: number;
  pendingAction: GameAction | null;
  onConfirm: () => void;
  onCancel: () => void;
  debugPolicyActions?: DebugActionEntry[] | null;
}) {
  const [hoveredPolicy, setHoveredPolicy] = useState<HoveredPolicy | null>(
    null,
  );
  const positions = Object.fromEntries(
    Object.values(state.nodes).map((n) => {
      const [x, y] = tilePixelVector(n.tile_coordinate, SIZE, 0, 0);
      const [dx, dy] = getNodeDelta(n.direction, SIZE * SQRT3, SIZE * 2);
      return [n.id, [x + dx, y + dy]];
    }),
  );
  const legal = disabled ? [] : state.current_playable_actions;
  const nodeActions = legal
    .filter((a) => a[1] === "BUILD_SETTLEMENT" || a[1] === "BUILD_CITY")
    .filter((a) => state.is_initial_build_phase || !mode || a[1] === mode);
  const roadActions = legal
    .filter((a) => a[1] === "BUILD_ROAD")
    .filter(
      () =>
        state.is_initial_build_phase ||
        !mode ||
        state.is_road_building ||
        mode === "BUILD_ROAD" ||
        state.current_prompt === "BUILD_ROAD",
    );
  const robberActions = legal.filter((a) => a[1] === "MOVE_ROBBER");
  const key = (e: React.KeyboardEvent, fn: () => void) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fn();
    }
  };

  // Partition debug board policy actions
  const policyRoads = (debugPolicyActions ?? []).filter(
    (e) => e.action[1] === "BUILD_ROAD",
  );
  const policySettlements = (debugPolicyActions ?? []).filter(
    (e) => e.action[1] === "BUILD_SETTLEMENT",
  );
  const policyCities = (debugPolicyActions ?? []).filter(
    (e) => e.action[1] === "BUILD_CITY",
  );
  const policyRobberList = (debugPolicyActions ?? []).filter(
    (e) => e.action[1] === "MOVE_ROBBER",
  );

  // Aggregate robber actions by destination tile coordinate (multiple victims may share a hex)
  const policyRobberByCoord = new Map<
    string,
    { coord: TileCoordinate; policy: number; entries: DebugActionEntry[] }
  >();
  for (const entry of policyRobberList) {
    const coord = (entry.action[2] as [TileCoordinate, string?])[0];
    const k = coord.join(",");
    const existing = policyRobberByCoord.get(k);
    if (existing) {
      existing.policy += entry.policy;
      existing.entries.push(entry);
    } else {
      policyRobberByCoord.set(k, {
        coord,
        policy: entry.policy,
        entries: [entry],
      });
    }
  }

  const hasBoardPolicyOverlay =
    policyRoads.length > 0 ||
    policySettlements.length > 0 ||
    policyCities.length > 0 ||
    policyRobberByCoord.size > 0;

  const maxBoardPolicy = Math.max(
    0,
    ...policyRoads.map((e) => e.policy),
    ...policySettlements.map((e) => e.policy),
    ...policyCities.map((e) => e.policy),
    ...Array.from(policyRobberByCoord.values()).map((r) => r.policy),
  );

  return (
    <svg
      className="island-board"
      viewBox="-345 -300 690 600"
      aria-label="Catan game board"
      data-testid="board"
      onMouseLeave={() => setHoveredPolicy(null)}
    >
      <defs>
        {Object.entries(TERRAIN).map(([r, c]) => (
          <linearGradient
            id={`terrain-${r}`}
            key={r}
            x1="0"
            x2="1"
            y1="0"
            y2="1"
          >
            <stop stopColor={c[0]} />
            <stop offset="1" stopColor={c[1]} />
          </linearGradient>
        ))}
        <filter id="piece-shadow" x="-70%" y="-70%" width="240%" height="240%">
          <feDropShadow dx="0" dy="2" stdDeviation="1.5" floodOpacity=".3" />
        </filter>
        <radialGradient id="placement">
          <stop stopColor="#f9ffdc" stopOpacity=".8" />
          <stop offset=".6" stopColor="#d9e98a" stopOpacity=".5" />
          <stop offset="1" stopColor="#789b2d" stopOpacity=".7" />
        </radialGradient>
      </defs>
      <g transform={`scale(${zoom})`}>
        <g
          className={`board-base-layer ${hasBoardPolicyOverlay ? "dimmed" : ""}`}
          data-testid="board-base-layer"
        >
          {state.tiles.map(({ tile, coordinate }) => {
            const [x, y] = tilePixelVector(coordinate, SIZE, 0, 0);
            if (tile.type === "WATER") return null;
            if (tile.type === "PORT")
              return (
                <g
                  key={`port-${coordinate}`}
                  transform={`translate(${x * 0.9} ${y * 0.9})`}
                  className="port-ship"
                >
                  <title>
                    {tile.resource
                      ? `2:1 ${tile.resource} port`
                      : "3:1 any resource port"}
                  </title>
                  <ellipse cy="29" rx="24" ry="5" fill="#64c0dd" />
                  <path
                    d="M-23 13h46l-8 14h-29Z"
                    fill="#be873e"
                    stroke="#71562e"
                    strokeWidth="1.4"
                  />
                  <path
                    d="M-21 16q21 13 42 0"
                    fill="none"
                    stroke="#f0c780"
                    strokeWidth="2"
                  />
                  <path d="M1-32v47" stroke="#8d6435" strokeWidth="3" />
                  <path
                    d="M-13-26Q-30-1-17 14q18 9 36-2Q10-8 16-26Z"
                    fill="#fff8e7"
                    stroke="#b4c2bd"
                    strokeWidth="1.3"
                  />
                  {tile.resource ? (
                    <g transform="translate(-8 -23) scale(.25)">
                      <Artwork name={tile.resource} />
                    </g>
                  ) : (
                    <text
                      y="-10"
                      textAnchor="middle"
                      fontSize="19"
                      fontWeight="800"
                      fill="#86664c"
                    >
                      ?
                    </text>
                  )}
                  <text
                    y="8"
                    textAnchor="middle"
                    fontSize="13"
                    fontWeight="700"
                    fill="#6e6758"
                  >
                    {tile.resource ? "2:1" : "3:1"}
                  </text>
                  <path d="M3-32q10-9 22-2L3-29" fill="#fff9e9" />
                </g>
              );
            const r = tile.type === "DESERT" ? "DESERT" : tile.resource;
            const moves = robberActions.filter((a) =>
              a[2][0].every((v, i) => v === coordinate[i]),
            );
            return (
              <g
                key={coordinate.join()}
                transform={`translate(${x} ${y})`}
                className={`terrain ${moves.length ? "robber-target" : ""}`}
                role={moves.length ? "button" : undefined}
                tabIndex={moves.length ? 0 : undefined}
                aria-label={
                  moves.length
                    ? `Move robber to ${tile.type === "RESOURCE_TILE" ? `${tile.number} ${r}` : "desert"}`
                    : undefined
                }
                onClick={moves.length ? () => onRobber(moves) : undefined}
                onKeyDown={(e) =>
                  moves.length > 0 && key(e, () => onRobber(moves))
                }
              >
                <title>
                  {tile.type === "DESERT" ? "Desert" : `${tile.number} · ${r}`}
                </title>
                <polygon
                  points={hex(66)}
                  fill="#63bdd4"
                  stroke="#8ed6e3"
                  strokeWidth="4"
                />
                <polygon
                  points={hex(63)}
                  fill="#f7e4b6"
                  stroke="#e6ce94"
                  strokeWidth="3"
                />
                <polygon
                  points={hex(58)}
                  fill={`url(#terrain-${r})`}
                  stroke="#9b864b"
                  strokeWidth="1.4"
                />
                <polygon
                  points={hex(54)}
                  fill="none"
                  stroke="#fffaab"
                  strokeOpacity=".45"
                  strokeWidth="1"
                />
                <g transform="translate(-19 -46) scale(.6)">
                  <Artwork name={r} />
                </g>
                <path
                  d="m-37 6 4-5 4 5m-7 2 1-7m65 7 4-5 4 5m-7 2 1-7"
                  fill="none"
                  stroke="#52704b"
                  opacity=".22"
                  strokeWidth="1.5"
                />
                {tile.type === "RESOURCE_TILE" && (
                  <g
                    transform="translate(0 23)"
                    className={
                      tile.number === 6 || tile.number === 8 ? "hot-number" : ""
                    }
                  >
                    <rect
                      x="-19"
                      y="-19"
                      width="38"
                      height="41"
                      rx="7"
                      fill="#fff9e9"
                      stroke="#8b8b73"
                      strokeWidth=".8"
                    />
                    <text textAnchor="middle" y="6" className="hex-number">
                      {tile.number}
                    </text>
                    <text textAnchor="middle" y="16" className="hex-pips">
                      {"•".repeat(6 - Math.abs(7 - tile.number))}
                    </text>
                  </g>
                )}
                {moves.length > 0 && (
                  <polygon
                    className="robber-outline"
                    points={hex(55)}
                    fill="none"
                    stroke="#fff6b7"
                    strokeWidth="4"
                  />
                )}
              </g>
            );
          })}
          {state.edges.map((e) => {
            const a = positions[e.id[0]],
              b = positions[e.id[1]];
            if (!a || !b) return null;
            // A filled hit area gives vertical roads a real width for touch,
            // keyboard focus, and browser hit testing (SVG line bounds are zero).
            const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
            const nx = (-(b[1] - a[1]) / length) * 12;
            const ny = ((b[0] - a[0]) / length) * 12;
            const action = roadActions.find(
              (a) => a[2].includes(e.id[0]) && a[2].includes(e.id[1]),
            );
            if (!e.color && !action) return null;
            return (
              <g
                key={e.id.join()}
                className={
                  action
                    ? `legal-road ${!mode && !state.is_initial_build_phase && !state.is_road_building ? "hover-placement" : ""}`
                    : "built-road"
                }
                role={action ? "button" : undefined}
                tabIndex={action ? 0 : undefined}
                aria-label={
                  action ? `Build road ${e.id.join(" to ")}` : undefined
                }
                onClick={action ? () => onAction(action) : undefined}
                onKeyDown={(ev) => action && key(ev, () => onAction(action))}
              >
                <line
                  x1={a[0]}
                  y1={a[1]}
                  x2={b[0]}
                  y2={b[1]}
                  stroke={e.color ? "#243d50" : "#f9e8ae"}
                  strokeWidth={e.color ? 12 : 15}
                  strokeLinecap="round"
                />
                <line
                  x1={a[0]}
                  y1={a[1]}
                  x2={b[0]}
                  y2={b[1]}
                  stroke={e.color ? COLOR_HEX[e.color] : "#a8bb54"}
                  strokeWidth="8"
                  strokeLinecap="round"
                  strokeDasharray={action ? "5 5" : undefined}
                />
                <line
                  x1={a[0]}
                  y1={a[1] - 2}
                  x2={b[0]}
                  y2={b[1] - 2}
                  stroke="white"
                  strokeWidth="1.3"
                  opacity=".3"
                />
                {action && (
                  <polygon
                    points={`${a[0] + nx},${a[1] + ny} ${b[0] + nx},${b[1] + ny} ${b[0] - nx},${b[1] - ny} ${a[0] - nx},${a[1] - ny}`}
                    fill="transparent"
                  />
                )}
              </g>
            );
          })}
          {Object.values(state.nodes).map((n) => {
            const [x, y] = positions[n.id];
            const a = nodeActions.find((a) => a[2] === n.id);
            if (!n.building && !a) return null;
            return (
              <g
                key={n.id}
                transform={`translate(${x} ${y})`}
                className={
                  a
                    ? `legal-node ${!mode && !state.is_initial_build_phase ? "hover-placement" : ""}`
                    : "built-node"
                }
                role={a ? "button" : undefined}
                tabIndex={a ? 0 : undefined}
                aria-label={
                  a
                    ? `Build ${a[1] === "BUILD_CITY" ? "city" : "settlement"} at intersection ${n.id}`
                    : undefined
                }
                onClick={a ? () => onAction(a) : undefined}
                onKeyDown={(e) => a && key(e, () => onAction(a))}
              >
                {a && (
                  <circle
                    r="12"
                    fill="url(#placement)"
                    stroke="#69852f"
                    strokeWidth="1.4"
                  />
                )}
                {n.building && (
                  <g
                    transform="translate(-16 -24) scale(.5)"
                    style={{ color: n.color ? COLOR_HEX[n.color] : "#fff" }}
                    filter="url(#piece-shadow)"
                  >
                    <Artwork
                      name={n.building === "CITY" ? "city" : "settlement"}
                    />
                  </g>
                )}
                {a && <circle r="16" fill="transparent" />}
              </g>
            );
          })}
          {(() => {
            const [x, y] = tilePixelVector(state.robber_coordinate, SIZE, 0, 0);
            return (
              <g
                transform={`translate(${x - 42} ${y - 8}) scale(.5)`}
                filter="url(#piece-shadow)"
              >
                <Artwork name="robber" />
              </g>
            );
          })()}
        </g>

        {hasBoardPolicyOverlay && (
          <g
            className="policy-overlay-layer"
            data-testid="policy-overlay-layer"
          >
            {/* Robber hex policy overlays */}
            {Array.from(policyRobberByCoord.entries()).map(
              ([coordKey, group]) => {
                const [x, y] = tilePixelVector(group.coord, SIZE, 0, 0);
                const t = policyIntensity(group.policy, maxBoardPolicy);
                const fill = policyColor(t, 0.25, 0.78);
                const stroke = policyColor(t, 0.65, 1.0);
                const tileTitle = describeTile(state, group.coord);
                const isSelected = group.entries.some((e) => e.selected);
                return (
                  <g
                    key={`policy-robber-${coordKey}`}
                    transform={`translate(${x} ${y})`}
                    className="policy-area policy-robber"
                    tabIndex={0}
                    aria-label={`Move robber to ${tileTitle}: ${formatPct(group.policy)}`}
                    onMouseEnter={() =>
                      setHoveredPolicy({
                        x,
                        y,
                        title: `Move Robber · ${tileTitle}`,
                        policy: group.policy,
                        entries: group.entries,
                      })
                    }
                    onFocus={() =>
                      setHoveredPolicy({
                        x,
                        y,
                        title: `Move Robber · ${tileTitle}`,
                        policy: group.policy,
                        entries: group.entries,
                      })
                    }
                    onMouseLeave={() => setHoveredPolicy(null)}
                    onBlur={() => setHoveredPolicy(null)}
                  >
                    <polygon
                      points={hex(55)}
                      fill={fill}
                      stroke={isSelected ? "#ffffff" : stroke}
                      strokeWidth={isSelected ? 4.5 : 2.5}
                    />
                  </g>
                );
              },
            )}

            {/* Road build policy overlays */}
            {policyRoads.map((entry) => {
              const edge = entry.action[2] as [number, number];
              const a = positions[edge[0]],
                b = positions[edge[1]];
              if (!a || !b) return null;
              const t = policyIntensity(entry.policy, maxBoardPolicy);
              const color = policyColor(t, 0.45, 0.98);
              const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
              const nx = (-(b[1] - a[1]) / length) * 13;
              const ny = ((b[0] - a[0]) / length) * 13;
              const midX = (a[0] + b[0]) / 2;
              const midY = (a[1] + b[1]) / 2;
              const title = `Build Road · ${edge[0]}–${edge[1]}`;
              return (
                <g
                  key={`policy-road-${edge[0]}-${edge[1]}`}
                  className="policy-area policy-road"
                  tabIndex={0}
                  aria-label={`${title}: ${formatPct(entry.policy)}`}
                  onMouseEnter={() =>
                    setHoveredPolicy({
                      x: midX,
                      y: midY,
                      title,
                      policy: entry.policy,
                      entries: [entry],
                    })
                  }
                  onFocus={() =>
                    setHoveredPolicy({
                      x: midX,
                      y: midY,
                      title,
                      policy: entry.policy,
                      entries: [entry],
                    })
                  }
                  onMouseLeave={() => setHoveredPolicy(null)}
                  onBlur={() => setHoveredPolicy(null)}
                >
                  {entry.selected && (
                    <line
                      x1={a[0]}
                      y1={a[1]}
                      x2={b[0]}
                      y2={b[1]}
                      stroke="#ffffff"
                      strokeWidth={11 + 6 * t}
                      strokeLinecap="round"
                      opacity="0.9"
                    />
                  )}
                  <line
                    x1={a[0]}
                    y1={a[1]}
                    x2={b[0]}
                    y2={b[1]}
                    stroke={color}
                    strokeWidth={7 + 6 * t}
                    strokeLinecap="round"
                  />
                  <polygon
                    points={`${a[0] + nx},${a[1] + ny} ${b[0] + nx},${b[1] + ny} ${b[0] - nx},${b[1] - ny} ${a[0] - nx},${a[1] - ny}`}
                    fill="transparent"
                  />
                </g>
              );
            })}

            {/* Settlement build policy overlays */}
            {policySettlements.map((entry) => {
              const nodeId = entry.action[2] as number;
              const pt = positions[nodeId];
              if (!pt) return null;
              const [x, y] = pt;
              const t = policyIntensity(entry.policy, maxBoardPolicy);
              const color = policyColor(t, 0.45, 0.98);
              const r = 10 + 7 * t;
              const title = `Build Settlement · #${nodeId}`;
              return (
                <g
                  key={`policy-settle-${nodeId}`}
                  transform={`translate(${x} ${y})`}
                  className="policy-area policy-settlement"
                  tabIndex={0}
                  aria-label={`${title}: ${formatPct(entry.policy)}`}
                  onMouseEnter={() =>
                    setHoveredPolicy({
                      x,
                      y,
                      title,
                      policy: entry.policy,
                      entries: [entry],
                    })
                  }
                  onFocus={() =>
                    setHoveredPolicy({
                      x,
                      y,
                      title,
                      policy: entry.policy,
                      entries: [entry],
                    })
                  }
                  onMouseLeave={() => setHoveredPolicy(null)}
                  onBlur={() => setHoveredPolicy(null)}
                >
                  <circle
                    r={r}
                    fill={color}
                    stroke={entry.selected ? "#ffffff" : "#1a2b34"}
                    strokeWidth={entry.selected ? 3 : 1.5}
                  />
                  <circle r={Math.max(17, r + 4)} fill="transparent" />
                </g>
              );
            })}

            {/* City build policy overlays (distinct diamond marker) */}
            {policyCities.map((entry) => {
              const nodeId = entry.action[2] as number;
              const pt = positions[nodeId];
              if (!pt) return null;
              const [x, y] = pt;
              const t = policyIntensity(entry.policy, maxBoardPolicy);
              const color = policyColor(t, 0.5, 0.98);
              const s = 12 + 7 * t;
              const title = `Build City · #${nodeId}`;
              return (
                <g
                  key={`policy-city-${nodeId}`}
                  transform={`translate(${x} ${y})`}
                  className="policy-area policy-city"
                  tabIndex={0}
                  aria-label={`${title}: ${formatPct(entry.policy)}`}
                  onMouseEnter={() =>
                    setHoveredPolicy({
                      x,
                      y,
                      title,
                      policy: entry.policy,
                      entries: [entry],
                    })
                  }
                  onFocus={() =>
                    setHoveredPolicy({
                      x,
                      y,
                      title,
                      policy: entry.policy,
                      entries: [entry],
                    })
                  }
                  onMouseLeave={() => setHoveredPolicy(null)}
                  onBlur={() => setHoveredPolicy(null)}
                >
                  <polygon
                    points={`0,${-s} ${s},0 0,${s} ${-s},0`}
                    fill={color}
                    stroke={entry.selected ? "#ffffff" : "#1a2b34"}
                    strokeWidth={entry.selected ? 3 : 1.8}
                  />
                  <circle r={Math.max(19, s + 4)} fill="transparent" />
                </g>
              );
            })}

            {/* Hover popup tooltip with assigned probability */}
            {hoveredPolicy &&
              (() => {
                const boxW = 186;
                const boxH =
                  hoveredPolicy.entries.length > 1
                    ? 56 + hoveredPolicy.entries.length * 16
                    : 64;
                const tx = Math.max(
                  -335,
                  Math.min(335 - boxW, hoveredPolicy.x - boxW / 2),
                );
                const ty =
                  hoveredPolicy.y - boxH - 18 < -292
                    ? hoveredPolicy.y + 18
                    : hoveredPolicy.y - boxH - 18;
                const primary = hoveredPolicy.entries[0];
                return (
                  <foreignObject
                    x={tx}
                    y={ty}
                    width={boxW}
                    height={boxH}
                    className="policy-tooltip-wrap"
                    style={{ pointerEvents: "none" }}
                  >
                    <div
                      className="policy-tooltip"
                      role="tooltip"
                      data-testid="policy-tooltip"
                    >
                      <div className="policy-tooltip-title">
                        {hoveredPolicy.title}
                      </div>
                      <div className="policy-tooltip-prob">
                        Policy: <strong>{formatPct(hoveredPolicy.policy)}</strong>
                      </div>
                      {hoveredPolicy.entries.length === 1 && primary && (
                        <div className="policy-tooltip-meta">
                          {primary.visits > 0 && (
                            <span>{primary.visits} visits</span>
                          )}
                          {primary.prior !== null && (
                            <span>Prior {formatPct(primary.prior)}</span>
                          )}
                          {primary.mean_value !== null && (
                            <span>
                              Q {(primary.mean_value * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                      )}
                      {hoveredPolicy.entries.length > 1 && (
                        <div className="policy-tooltip-victims">
                          {hoveredPolicy.entries.map((e, idx) => {
                            const victim = (
                              e.action[2] as [TileCoordinate, string?]
                            )[1];
                            return (
                              <div key={idx}>
                                <span>{victim ? `Steal ${victim}` : "No steal"}</span>
                                <b>{formatPct(e.policy)}</b>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </foreignObject>
                );
              })()}
          </g>
        )}

        {pendingAction &&
          (() => {
            const isRoad = pendingAction[1] === "BUILD_ROAD";
            const isCity = pendingAction[1] === "BUILD_CITY";
            let point: number[] | undefined;
            if (isRoad) {
              const edge = pendingAction[2] as [number, number];
              const a = positions[edge[0]],
                b = positions[edge[1]];
              if (a && b) point = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
            } else point = positions[pendingAction[2] as number];
            if (!point) return null;
            const label = isRoad ? "road" : isCity ? "city" : "settlement";
            const free =
              state.is_initial_build_phase ||
              (state.is_road_building && isRoad);
            const cost = free ? [] : (BUILD_COST[pendingAction[1]] ?? []);
            return (
              <foreignObject
                x={Math.max(-330, Math.min(225, point[0] - 49))}
                y={Math.max(-290, point[1] - 124)}
                width="108"
                height="116"
                className="placement-confirmation"
              >
                <div
                  className="placement-popup"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div
                    className="placement-cost"
                    aria-label={free ? "Free placement" : "Building cost"}
                  >
                    {free ? (
                      <span>Free</span>
                    ) : (
                      cost.map((r, i) => <Icon key={i} name={r} />)
                    )}
                  </div>
                  <button
                    className="placement-accept action-tile"
                    onClick={onConfirm}
                    disabled={disabled}
                    aria-label={`Confirm ${label} placement`}
                  >
                    <Icon
                      name={label}
                      style={{ color: COLOR_HEX[state.current_color] }}
                    />
                    <span>Confirm</span>
                  </button>
                  <button
                    className="placement-cancel"
                    onClick={onCancel}
                    aria-label="Cancel placement"
                  >
                    <Icon name="close" />
                  </button>
                </div>
              </foreignObject>
            );
          })()}
      </g>
    </svg>
  );
}
