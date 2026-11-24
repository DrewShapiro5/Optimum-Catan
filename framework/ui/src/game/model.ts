import type { Color, GameState, ResourceCard } from "../utils/api.types";

export const RESOURCES: ResourceCard[] = [
  "WOOD",
  "BRICK",
  "SHEEP",
  "WHEAT",
  "ORE",
];
export const COLORS: Color[] = ["RED", "BLUE", "ORANGE", "WHITE"];
export const COLOR_HEX: Record<Color, string> = {
  RED: "#ef4547",
  BLUE: "#2865d7",
  ORANGE: "#eea333",
  WHITE: "#efe9d9",
};
export const LABELS: Record<ResourceCard, string> = {
  WOOD: "Lumber",
  BRICK: "Brick",
  SHEEP: "Wool",
  WHEAT: "Grain",
  ORE: "Ore",
};
export const DEV_CARDS = [
  "KNIGHT",
  "MONOPOLY",
  "YEAR_OF_PLENTY",
  "ROAD_BUILDING",
  "VICTORY_POINT",
] as const;

export function stat(state: GameState, color: Color, field: string): number {
  return Number(
    state.player_state[`P${state.colors.indexOf(color)}_${field}`] ?? 0,
  );
}

export function playerNames(gameId: string): Partial<Record<Color, string>> {
  try {
    return JSON.parse(
      localStorage.getItem(`catanatron.players.${gameId}`) ?? "{}",
    );
  } catch {
    return {};
  }
}

export function saveNames(gameId: string, names: string[]) {
  try {
    localStorage.setItem(
      `catanatron.players.${gameId}`,
      JSON.stringify(
        Object.fromEntries(
          names.map((name, i) => [COLORS[i], name.trim() || `Player ${i + 1}`]),
        ),
      ),
    );
  } catch {
    /* Gameplay also works when browser storage is unavailable. */
  }
}

export function turnPrompt(state: GameState): string {
  if (state.winning_color) return "Winner!";
  if (state.is_road_building)
    return `Place a free road · ${state.free_roads_available ?? 0} remaining`;
  if (state.is_initial_build_phase)
    return state.current_prompt.includes("SETTLEMENT")
      ? "Place a settlement"
      : "Place a road";
  if (state.current_prompt === "DISCARD")
    return `Discard ${state.current_discard_count} cards`;
  if (state.current_prompt === "MOVE_ROBBER") return "Move the robber";
  return state.current_playable_actions.some((a) => a[1] === "ROLL")
    ? "Roll the dice"
    : "Your turn";
}

export type PlayerProvider = { key: string; name: string; serverKey: string };
// API providers are discovered from the server's central /api/agents catalog.

export const HUMAN: PlayerProvider = {
  key: "human",
  name: "Human",
  serverKey: "HUMAN",
};

export const BUILD_COST: Record<string, ResourceCard[]> = {
  BUILD_ROAD: ["WOOD", "BRICK"],
  BUILD_SETTLEMENT: ["WOOD", "BRICK", "SHEEP", "WHEAT"],
  BUILD_CITY: ["WHEAT", "WHEAT", "ORE", "ORE", "ORE"],
};
