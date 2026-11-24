import type { GameAction, GameState, ResourceCard } from "../utils/api.types";
import { RESOURCES, stat } from "./model";
export const resourceCounts = (cards: ResourceCard[]) =>
  RESOURCES.map((r) => cards.filter((c) => c === r).length);
export const sameResources = (a: ResourceCard[], b: ResourceCard[]) =>
  resourceCounts(a).every((n, i) => n === resourceCounts(b)[i]);

/** Compose larger bank exchanges from the engine's legal single-card exchanges. */
export function bankTradePlan(
  state: GameState,
  give: ResourceCard[],
  take: ResourceCard[],
): GameAction[] | null {
  if (!give.length || !take.length) return null;
  const offers = state.current_playable_actions.filter(
    (a) => a[1] === "MARITIME_TRADE",
  );
  const outgoing = resourceCounts(give),
    incoming = resourceCounts(take);
  if (
    outgoing.some(
      (n, i) => n > stat(state, state.current_color, `${RESOURCES[i]}_IN_HAND`),
    )
  )
    return null;
  if (incoming.some((n, i) => n > state.resource_freqdeck[i])) return null;
  const failed = new Set<string>();
  function plan(out: number[], inc: number[]): GameAction[] | null {
    if (inc.every((n) => n === 0)) return out.every((n) => n === 0) ? [] : null;
    const key = `${out.join()}/${inc.join()}`;
    if (failed.has(key)) return null;
    for (const a of offers) {
      const resource = a[2][0];
      if (!resource) continue;
      const from = RESOURCES.indexOf(resource),
        to = RESOURCES.indexOf(a[2][4]!);
      const rate = a[2].slice(0, 4).filter(Boolean).length;
      if (out[from] < rate || inc[to] < 1) continue;
      const nextOut = [...out],
        nextIn = [...inc];
      nextOut[from] -= rate;
      nextIn[to]--;
      const rest = plan(nextOut, nextIn);
      if (rest) return [a, ...rest];
    }
    failed.add(key);
    return null;
  }
  return plan(outgoing, incoming);
}
export function plentyAction(state: GameState, selected: ResourceCard[]) {
  return state.current_playable_actions.find(
    (a) => a[1] === "PLAY_YEAR_OF_PLENTY" && sameResources(a[2], selected),
  );
}
export function canAddPlenty(
  state: GameState,
  selected: ResourceCard[],
  next: ResourceCard,
) {
  const counts = resourceCounts([...selected, next]);
  return state.current_playable_actions.some(
    (a) =>
      a[1] === "PLAY_YEAR_OF_PLENTY" &&
      counts.every((n, i) => n <= resourceCounts(a[2])[i]),
  );
}
