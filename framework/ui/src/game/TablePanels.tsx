import { useEffect, useRef } from "react";
import type { Color, GameActionRecord, GameState } from "../utils/api.types";
import Icon, { type IconName } from "./Icons";
import Card from "./Cards";
import { COLOR_HEX, DEV_CARDS, RESOURCES, stat } from "./model";

export function Dice({
  values,
  small = false,
}: {
  values?: number[];
  small?: boolean;
}) {
  const dots: Record<number, number[]> = {
    1: [4],
    2: [0, 8],
    3: [0, 4, 8],
    4: [0, 2, 6, 8],
    5: [0, 2, 4, 6, 8],
    6: [0, 2, 3, 5, 6, 8],
  };
  return (
    <span
      className={`dice-pair ${small ? "small" : ""}`}
      aria-label={
        values ? `Rolled ${values[0]} and ${values[1]}` : "Dice not rolled yet"
      }
    >
      {(values ?? [0, 0]).map((v, i) => (
        <span className="die" key={i}>
          {v === 0 ? (
            <span className="die-unrolled">–</span>
          ) : (
            Array.from({ length: 9 }, (_, j) => (
              <i key={j} className={dots[v]?.includes(j) ? "pip" : ""} />
            ))
          )}
        </span>
      ))}
    </span>
  );
}
export function PlayerPanel({
  state,
  color,
  name,
}: {
  state: GameState;
  color: Color;
  name: string;
}) {
  const prefix = `P${state.colors.indexOf(color)}_`;
  const resources = Number(state.player_state[prefix + "NUM_RESOURCES_IN_HAND"] ?? RESOURCES.reduce(
    (n, r) => n + stat(state, color, `${r}_IN_HAND`),
    0,
  ));
  const devs = Number(state.player_state[prefix + "NUM_DEVELOPMENT_CARDS_IN_HAND"] ?? DEV_CARDS.reduce(
    (n, r) => n + stat(state, color, `${r}_IN_HAND`),
    0,
  ));
  const active = state.current_color === color;
  return (
    <div
      className={`player-panel cream-panel ${active ? "active" : ""}`}
      style={{ "--player-color": COLOR_HEX[color] } as React.CSSProperties}
    >
      <div className="player-identity">
        <div className="avatar">
          <Icon name="person" />
          <span className="vp-ribbon" title="Victory Points">
            {stat(state, color, "VICTORY_POINTS")}
          </span>
        </div>
        <div>
          <strong>{name}</strong>
          <span>
            {color.toLowerCase()} ·{" "}
            {active
              ? "playing"
              : state.bot_colors.includes(color)
                ? "agent"
                : "human"}
          </span>
        </div>
      </div>
      <div className="player-numbers">
        <Card resource="hidden" count={resources} small />
        <Card resource="dev" count={devs} small />
        <span
          className={`player-award ${stat(state, color, "HAS_ARMY") ? "awarded" : ""}`}
          title="Knights played"
        >
          <Icon name="army" />
          <b>{stat(state, color, "PLAYED_KNIGHT")}</b>
        </span>
        <span
          className={`player-award ${stat(state, color, "HAS_ROAD") ? "awarded" : ""}`}
          title="Longest road"
        >
          <span className="road-award">⌁</span>
          <b>{stat(state, color, "LONGEST_ROAD_LENGTH")}</b>
        </span>
      </div>
    </div>
  );
}
function RecordContent({
  record,
  names,
}: {
  record: GameActionRecord;
  names: Partial<Record<Color, string>>;
}) {
  const [a, result] = record;
  const player = (
    <strong
      style={{
        color:
          a[0] === "WHITE"
            ? "#756b57"
            : a[0] === "ORANGE"
              ? "#a16617"
              : COLOR_HEX[a[0]],
      }}
    >
      {names[a[0]] ?? `${a[0][0]}${a[0].slice(1).toLowerCase()}`}
    </strong>
  );
  const pieces: Record<string, [string, IconName]> = {
    BUILD_SETTLEMENT: ["placed a settlement", "settlement"],
    BUILD_CITY: ["built a city", "city"],
    BUILD_ROAD: ["built a road", "road"],
    BUY_DEVELOPMENT_CARD: ["bought a development card", "dev"],
    PLAY_KNIGHT_CARD: ["played a knight", "army"],
    PLAY_ROAD_BUILDING: ["played road building", "road"],
  };
  if (a[1] === "ROLL")
    return (
      <>
        {player} rolled <Dice values={result as number[]} small />
      </>
    );
  if (pieces[a[1]]) {
    const [text, icon] = pieces[a[1]];
    return (
      <>
        {player} {text} <Icon name={icon} style={{ color: COLOR_HEX[a[0]] }} />
      </>
    );
  }
  if (a[1] === "MARITIME_TRADE")
    return (
      <>
        {player} traded{" "}
        {a[2]
          .slice(0, 4)
          .filter(Boolean)
          .map((r, i) => (
            <Card key={i} resource={r!} small />
          ))}{" "}
        for <Card resource={a[2][4]!} small />
      </>
    );
  if (a[1] === "DISCARD_RESOURCE")
    return (
      <>
        {player} discarded <Card resource={a[2]} small />
      </>
    );
  if (a[1] === "MOVE_ROBBER")
    return (
      <>
        {player} moved the robber <Icon name="robber" />
        {result && (
          <>
            {" "}
            and stole <Card resource={result as any} small />
          </>
        )}
      </>
    );
  if (a[1] === "END_TURN") return <>{player} ended their turn.</>;
  if (a[1] === "PLAY_MONOPOLY")
    return (
      <>
        {player} played Monopoly for <Card resource={a[2]} small />
      </>
    );
  if (a[1] === "PLAY_YEAR_OF_PLENTY")
    return (
      <>
        {player} received{" "}
        {a[2].map((r, i) => (
          <Card key={i} resource={r} small />
        ))}
      </>
    );
  return (
    <>
      {player} {a[1].replaceAll("_", " ").toLowerCase()}
    </>
  );
}
export function GameLog({
  state,
  names,
}: {
  state: GameState;
  names: Partial<Record<Color, string>>;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [state.state_index]);
  return (
    <section className="game-log cream-panel">
      <div className="panel-title">
        <span>GAME LOG</span>
        <span>Turn {state.num_turns + 1}</span>
      </div>
      <div
        className="log-scroll"
        ref={ref}
        role="log"
        aria-label="Game history"
      >
        <p className="log-welcome">
          Place two settlements and two roads to begin.
        </p>
        {state.action_records.map((r, i) => (
          <div
            className={`log-entry ${r[0][1] === "ROLL" ? "new-roll" : ""}`}
            key={i}
          >
            <Icon name="person" />
            <span>
              <RecordContent record={r} names={names} />
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
export function Bank({ state }: { state: GameState }) {
  return (
    <section className="bank-panel cream-panel" aria-label="Bank">
      <div className="panel-title">
        <span>
          <Icon name="bank" />
          BANK
        </span>
        <span>Cards remaining</span>
      </div>
      <div className="bank-cards">
        {RESOURCES.map((r, i) => (
          <Card resource={r} count={state.resource_freqdeck[i]} key={r} small />
        ))}
        <Card
          resource="dev"
          count={Object.values(state.development_listdeck).reduce(
            (n, c) => n + c,
            0,
          )}
          small
        />
      </div>
    </section>
  );
}
