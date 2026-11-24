import { expect, test } from "vitest";
import { bankTradePlan, canAddPlenty, plentyAction } from "./resourceSelection";
import { makeGameState } from "../test/fixtures";
import type { GameAction } from "../utils/api.types";
const woodOre: GameAction = [
  "BLUE",
  "MARITIME_TRADE",
  ["WOOD", "WOOD", "WOOD", "WOOD", "ORE"],
];
const sheepOre: GameAction = [
  "BLUE",
  "MARITIME_TRADE",
  ["SHEEP", "SHEEP", null, null, "ORE"],
];
const state = makeGameState({
  player_state: { P1_WOOD_IN_HAND: 8, P1_SHEEP_IN_HAND: 4 },
  current_playable_actions: [woodOre, sheepOre],
});
test("composes multiple bank trades at engine-provided normal and port rates", () => {
  expect(
    bankTradePlan(
      state,
      ["WOOD", "WOOD", "WOOD", "WOOD", "SHEEP", "SHEEP"],
      ["ORE", "ORE"],
    ),
  ).toEqual([woodOre, sheepOre]);
  expect(bankTradePlan(state, Array(8).fill("WOOD"), ["ORE", "ORE"])).toEqual([
    woodOre,
    woodOre,
  ]);
});
test("rejects incorrect ratios, empty trades, insufficient hand and bank stock", () => {
  expect(bankTradePlan(state, ["WOOD", "WOOD", "WOOD"], ["ORE"])).toBeNull();
  expect(bankTradePlan(state, [], [])).toBeNull();
  expect(
    bankTradePlan(state, Array(12).fill("WOOD"), ["ORE", "ORE", "ORE"]),
  ).toBeNull();
  expect(
    bankTradePlan(
      { ...state, resource_freqdeck: [19, 19, 19, 19, 0] },
      Array(4).fill("WOOD"),
      ["ORE"],
    ),
  ).toBeNull();
});
test("Year of Plenty uses only legal combinations, including two of the same resource", () => {
  const a: GameAction = ["BLUE", "PLAY_YEAR_OF_PLENTY", ["ORE", "ORE"]];
  const game = { ...state, current_playable_actions: [a] };
  expect(canAddPlenty(game, [], "ORE")).toBe(true);
  expect(canAddPlenty(game, ["ORE"], "ORE")).toBe(true);
  expect(canAddPlenty(game, ["ORE", "ORE"], "ORE")).toBe(false);
  expect(canAddPlenty(game, [], "WOOD")).toBe(false);
  expect(plentyAction(game, ["ORE"])).toBeUndefined();
  expect(plentyAction(game, ["ORE", "ORE"])).toEqual(a);
});
