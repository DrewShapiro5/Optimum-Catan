import { useState } from "react";
import type { GameAction, GameState, ResourceCard } from "../utils/api.types";
import Icon from "./Icons";
import Card from "./Cards";
import { RESOURCES, stat } from "./model";
import { bankTradePlan, canAddPlenty, plentyAction } from "./resourceSelection";

export type ResourceMode = "trade" | "plenty" | "discard";
export default function ResourcePanel({
  state,
  mode,
  give,
  setGive,
  busy,
  onSubmit,
  onClose,
}: {
  state: GameState;
  mode: ResourceMode;
  give: ResourceCard[];
  setGive: (cards: ResourceCard[]) => void;
  busy: boolean;
  onSubmit: (actions: GameAction[]) => void;
  onClose: () => void;
}) {
  const [take, setTake] = useState<ResourceCard[]>([]);
  const color = state.current_color;
  const plan = mode === "trade" ? bankTradePlan(state, give, take) : null;
  const plenty = mode === "plenty" ? plentyAction(state, take) : undefined;
  const needed =
    mode === "discard"
      ? state.current_discard_count
      : Math.max(
          0,
          ...state.current_playable_actions
            .filter((a) => a[1] === "PLAY_YEAR_OF_PLENTY")
            .map((a) => (a[2] as ResourceCard[]).length),
        );
  const valid =
    mode === "trade"
      ? !!plan
      : mode === "plenty"
        ? !!plenty
        : give.length === needed;
  function confirm() {
    if (!valid || busy) return;
    const actions =
      mode === "trade"
        ? plan!
        : mode === "plenty"
          ? [plenty!]
          : give.map((r) => [color, "DISCARD_RESOURCE", r] as GameAction);
    onSubmit(actions);
  }
  const canTake = (r: ResourceCard, i: number) =>
    !busy &&
    (mode === "plenty"
      ? canAddPlenty(state, take, r)
      : take.filter((c) => c === r).length < state.resource_freqdeck[i] &&
        take.length <
          Math.max(
            1,
            Math.floor(
              RESOURCES.reduce(
                (n, r) => n + stat(state, color, `${r}_IN_HAND`),
                0,
              ) / 2,
            ),
          ));
  return (
    <section
      className={`resource-composer cream-panel composer-${mode}`}
      role="dialog"
      aria-label={
        mode === "trade"
          ? "Trade resources"
          : mode === "plenty"
            ? "Year of Plenty"
            : "Discard cards"
      }
    >
      <div className="composer-heading">
        <strong>
          {mode === "trade"
            ? "Trade resources"
            : mode === "plenty"
              ? `Year of Plenty (${take.length}/${needed})`
              : `Discard cards (${give.length}/${needed})`}
        </strong>
        <button
          className="composer-close"
          onClick={onClose}
          disabled={busy}
          aria-label="Close resource panel"
        >
          <Icon name="close" />
        </button>
      </div>
      {mode !== "discard" && (
        <div className="composer-bank">
          <div
            className="composer-resource-list"
            aria-label="Resources to receive"
          >
            {RESOURCES.map((r, i) => (
              <Card
                key={r}
                resource={r}
                count={state.resource_freqdeck[i]}
                disabled={!canTake(r, i)}
                onClick={() => setTake([...take, r])}
              />
            ))}
          </div>
          <Icon name="bank" />
        </div>
      )}
      <div className="composer-body">
        <div className="composer-selections">
          {mode !== "discard" && (
            <div className="composer-row">
              <span className="selection-direction receive">
                <Icon name="army" />
                <b>↓</b>
                <small>Receive</small>
              </span>
              <div
                className="selected-cards"
                aria-label="Selected resources to receive"
              >
                {take.map((r, i) => (
                  <Card
                    key={i}
                    resource={r}
                    small
                    disabled={busy}
                    onClick={() => setTake(take.filter((_, j) => j !== i))}
                  />
                ))}
                {!take.length && (
                  <span className="selection-hint">Select resources above</span>
                )}
              </div>
            </div>
          )}
          {mode !== "plenty" && (
            <div className="composer-row">
              <span className="selection-direction give">
                <Icon name="person" />
                <b>↑</b>
                <small>{mode === "discard" ? "Discard" : "Offer"}</small>
              </span>
              <div
                className="selected-cards"
                aria-label="Selected resources from hand"
              >
                {give.map((r, i) => (
                  <Card
                    key={i}
                    resource={r}
                    small
                    disabled={busy}
                    onClick={() => setGive(give.filter((_, j) => j !== i))}
                  />
                ))}
                {!give.length && (
                  <span className="selection-hint">
                    {mode === "discard"
                      ? `Select ${needed} cards from your hand`
                      : "Select cards from your hand"}
                  </span>
                )}
              </div>
            </div>
          )}
          {mode === "discard" && (
            <p className="discard-explanation">
              7 was rolled. Discard {needed} cards. Hands above{" "}
              {state.game.discard_limit} cards discard half.
            </p>
          )}
        </div>
        <div className="composer-buttons">
          <button
            className="action-tile"
            disabled={!valid || busy}
            onClick={confirm}
            aria-label={
              mode === "trade"
                ? "Confirm bank trade"
                : mode === "plenty"
                  ? "Confirm resources"
                  : "Confirm discards"
            }
          >
            <Icon name={mode === "discard" ? "check" : "bank"} />
            <span>{mode === "trade" ? "Bank" : "Confirm"}</span>
          </button>
          {mode === "trade" && (
            <button
              className="action-tile"
              disabled
              aria-label="Player trade unavailable"
              title="Player trading is not enabled"
            >
              <Icon name="army" />
              <span>Player</span>
            </button>
          )}
          <button
            className="action-tile"
            disabled={busy}
            onClick={onClose}
            aria-label="Cancel resource selection"
          >
            <Icon name="close" />
            <span>Cancel</span>
          </button>
        </div>
      </div>
    </section>
  );
}
