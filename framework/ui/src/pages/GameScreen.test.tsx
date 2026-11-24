import { useContext } from "react";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { StateProvider, store } from "../store";
import { getState, postAction, setPerspective } from "../utils/apiClient";
import { makeGameState } from "../test/fixtures";
import GameScreen from "./GameScreen";
vi.mock("../utils/apiClient", () => ({
  getState: vi.fn(),
  postAction: vi.fn(),
  setPerspective: vi.fn(),
}));
beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});
afterEach(cleanup);
function Probe() {
  const { state } = useContext(store);
  return (
    <output data-testid="state-index">{state.gameState?.state_index}</output>
  );
}
function show(replay = false) {
  render(
    <StateProvider>
      <MemoryRouter
        initialEntries={[
          replay ? "/games/game-123/states/7" : "/games/game-123",
        ]}
      >
        <Routes>
          <Route
            path={
              replay ? "/games/:gameId/states/:stateIndex" : "/games/:gameId"
            }
            element={<GameScreen replayMode={replay} />}
          />
        </Routes>
      </MemoryRouter>
      <Probe />
    </StateProvider>,
  );
}
test("keeps the human hand visible while the opponent searches", async () => {
  const game = makeGameState({current_color: "RED", viewer_color: "BLUE", current_playable_actions: []});
  game.player_state.P1_WHEAT_IN_HAND = 3;
  game.player_state.P1_WOOD_IN_HAND = 0;
  game.player_state.P0_KNIGHT_IN_HAND = 2; // Defensive rendering even if a payload is stale.
  game.player_state.P0_VICTORY_POINT_IN_HAND = 1;
  vi.mocked(getState).mockResolvedValue(game);
  vi.mocked(postAction).mockReturnValue(new Promise(() => {}));
  show();
  const hand = await screen.findByRole("region", {name: "Current player's hand"});
  expect(within(hand).getByText(/Blue’s hand/)).toBeInTheDocument();
  expect(within(hand).getByText(/3 resource cards/)).toBeInTheDocument();
  expect(within(hand).queryByText("KNIGHT")).not.toBeInTheDocument();
  expect(within(hand).queryByText("VICTORY POINT")).not.toBeInTheDocument();
  expect(screen.getAllByText("Private hand").length).toBeGreaterThan(0);
});
test("loads saved games, renders bank and hand, and submits the active human's roll", async () => {
  vi.mocked(getState).mockResolvedValue(makeGameState());
  vi.mocked(postAction).mockResolvedValue(
    makeGameState({
      state_index: 13,
      current_playable_actions: [["BLUE", "END_TURN", null]],
    }),
  );
  show();
  expect(await screen.findByTestId("board")).toBeInTheDocument();
  expect(setPerspective).toHaveBeenCalledWith(null);
  expect(screen.getByRole("region", { name: "Bank" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Roll dice" }));
  expect(postAction).toHaveBeenCalledWith("game-123", ["BLUE", "ROLL", null]);
  expect(screen.getByTestId("state-index")).toHaveTextContent("13");
  expect(screen.getByRole("button", { name: "End turn" })).toBeEnabled();
});
test("posts legal board placement and hands control to the next human", async () => {
  vi.mocked(getState).mockResolvedValue(
    makeGameState({
      bot_colors: [],
      current_color: "RED",
      is_initial_build_phase: true,
      current_prompt: "BUILD_INITIAL_SETTLEMENT",
      current_playable_actions: [["RED", "BUILD_SETTLEMENT", 0]],
    }),
  );
  vi.mocked(postAction).mockResolvedValue(
    makeGameState({
      bot_colors: [],
      current_color: "BLUE",
      current_playable_actions: [["BLUE", "ROLL", null]],
      state_index: 13,
    }),
  );
  show();
  await userEvent.click(
    await screen.findByRole("button", {
      name: "Build settlement at intersection 0",
    }),
  );
  expect(postAction).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Confirm settlement placement" }),
  );
  expect(postAction).toHaveBeenCalledWith("game-123", [
    "RED",
    "BUILD_SETTLEMENT",
    0,
  ]);
  expect(
    screen.getByRole("status", { name: "Current turn" }),
  ).toHaveTextContent("BLUE'S TURN");
  expect(screen.getByRole("button", { name: "Roll dice" })).toBeEnabled();
});
test("persisted bot games still advance through the backend", async () => {
  vi.mocked(getState).mockResolvedValue(
    makeGameState({ current_color: "RED" }),
  );
  vi.mocked(postAction).mockResolvedValue(makeGameState({ state_index: 13 }));
  show();
  await waitFor(() =>
    expect(screen.getByTestId("state-index")).toHaveTextContent("13"),
  );
  expect(postAction).toHaveBeenCalledWith("game-123", undefined);
});
test("replays load historical state and cannot post moves", async () => {
  vi.mocked(getState).mockResolvedValue(makeGameState({ state_index: 7 }));
  show(true);
  await screen.findByTestId("board");
  expect(getState).toHaveBeenCalledWith("game-123", "7");
  expect(screen.getByRole("button", { name: "Roll dice" })).toBeDisabled();
  expect(postAction).not.toHaveBeenCalled();
});
test("bank trades submit the exact legal exchange", async () => {
  const trade = [
    "BLUE",
    "MARITIME_TRADE",
    ["WOOD", "WOOD", "WOOD", "WOOD", "ORE"],
  ] as const;
  vi.mocked(getState).mockResolvedValue(
    makeGameState({
      current_playable_actions: [trade as any],
      player_state: { P1_WOOD_IN_HAND: 4 },
    }),
  );
  vi.mocked(postAction).mockResolvedValue(makeGameState({ state_index: 13 }));
  show();
  await userEvent.click(await screen.findByRole("button", { name: "Trade" }));
  const dialog = screen.getByRole("dialog", { name: "Trade resources" });
  expect(dialog).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Lumber: 4" }));
  await userEvent.click(screen.getByRole("button", { name: "Ore: 19" }));
  await userEvent.click(
    screen.getByRole("button", { name: "Confirm bank trade" }),
  );
  expect(postAction).toHaveBeenCalledWith("game-123", trade);
});
test("discard requires the correct quantity and submits selected resources sequentially", async () => {
  const before = makeGameState({
    current_prompt: "DISCARD",
    current_discard_count: 2,
    player_state: { P1_WOOD_IN_HAND: 3, P1_BRICK_IN_HAND: 2 },
    current_playable_actions: [
      ["BLUE", "DISCARD_RESOURCE", "WOOD"],
      ["BLUE", "DISCARD_RESOURCE", "BRICK"],
    ],
  });
  vi.mocked(getState).mockResolvedValue(before);
  vi.mocked(postAction)
    .mockResolvedValueOnce({
      ...before,
      state_index: 13,
      current_discard_count: 1,
    })
    .mockResolvedValueOnce(makeGameState({ state_index: 14 }));
  show();
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: "Discard" }));
  expect(
    screen.getByRole("button", { name: "Confirm discards" }),
  ).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Lumber: 3" }));
  await user.click(screen.getByRole("button", { name: "Brick: 2" }));
  await user.click(screen.getByRole("button", { name: "Confirm discards" }));
  expect(postAction).toHaveBeenNthCalledWith(1, "game-123", [
    "BLUE",
    "DISCARD_RESOURCE",
    "WOOD",
  ]);
  expect(postAction).toHaveBeenNthCalledWith(2, "game-123", [
    "BLUE",
    "DISCARD_RESOURCE",
    "BRICK",
  ]);
  expect(screen.getByTestId("state-index")).toHaveTextContent("14");
});
test("a failed action shows an error and reloads authoritative state", async () => {
  vi.mocked(getState)
    .mockResolvedValueOnce(makeGameState())
    .mockResolvedValueOnce(makeGameState({ state_index: 20 }));
  vi.mocked(postAction).mockRejectedValue(new Error("stale state"));
  show();
  await userEvent.click(
    await screen.findByRole("button", { name: "Roll dice" }),
  );
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "couldn't be completed",
  );
  await waitFor(() =>
    expect(screen.getByTestId("state-index")).toHaveTextContent("20"),
  );
});

test("road-building cards immediately expose the free legal roads", async () => {
  vi.mocked(getState).mockResolvedValue(
    makeGameState({
      is_road_building: true,
      free_roads_available: 2,
      nodes: {
        ...makeGameState().nodes,
        "1": {
          id: 1,
          tile_coordinate: [0, 0, 0],
          direction: "NORTHWEST",
          building: null,
          color: null,
        },
      },
      current_playable_actions: [["BLUE", "BUILD_ROAD", [0, 1]]],
    }),
  );
  vi.mocked(postAction).mockResolvedValue(makeGameState({ state_index: 13 }));
  show();
  await userEvent.click(
    await screen.findByRole("button", { name: "Build road 0 to 1" }),
  );
  expect(postAction).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Confirm road placement" }),
  );
  expect(postAction).toHaveBeenCalledWith("game-123", [
    "BLUE",
    "BUILD_ROAD",
    [0, 1],
  ]);
});

test("trade opens even without an affordable exchange, with both submit buttons disabled", async () => {
  vi.mocked(getState).mockResolvedValue(makeGameState());
  show();
  await userEvent.click(await screen.findByRole("button", { name: "Trade" }));
  expect(
    screen.getByRole("dialog", { name: "Trade resources" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Confirm bank trade" }),
  ).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Player trade unavailable" }),
  ).toBeDisabled();
});
test("Year of Plenty opens the hand panel and confirms exactly two resources", async () => {
  const action = ["BLUE", "PLAY_YEAR_OF_PLENTY", ["ORE", "ORE"]] as any;
  vi.mocked(getState).mockResolvedValue(
    makeGameState({
      current_playable_actions: [action],
      player_state: { P1_YEAR_OF_PLENTY_IN_HAND: 1 },
    }),
  );
  vi.mocked(postAction).mockResolvedValue(makeGameState({ state_index: 13 }));
  show();
  const user = userEvent.setup();
  await user.click(await screen.findByTitle("YEAR OF PLENTY"));
  await user.click(screen.getByRole("button", { name: /Year of Plenty/ }));
  expect(
    screen.getByRole("dialog", { name: "Year of Plenty" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Confirm resources" }),
  ).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Ore: 19" }));
  expect(
    screen.getByRole("button", { name: "Confirm resources" }),
  ).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Ore: 19" }));
  await user.click(screen.getByRole("button", { name: "Confirm resources" }));
  expect(postAction).toHaveBeenCalledWith("game-123", action);
});
test("canceling a placement leaves the game unchanged", async () => {
  vi.mocked(getState).mockResolvedValue(
    makeGameState({
      bot_colors: [],
      current_color: "RED",
      is_initial_build_phase: true,
      current_playable_actions: [["RED", "BUILD_SETTLEMENT", 0]],
    }),
  );
  show();
  await userEvent.click(
    await screen.findByRole("button", {
      name: "Build settlement at intersection 0",
    }),
  );
  expect(
    screen.getByRole("button", { name: "Confirm settlement placement" }),
  ).toBeInTheDocument();
  await userEvent.click(
    screen.getByRole("button", { name: "Cancel placement" }),
  );
  expect(
    screen.queryByRole("button", { name: "Confirm settlement placement" }),
  ).not.toBeInTheDocument();
  expect(postAction).not.toHaveBeenCalled();
});

test.each([2, 3, 4])(
  "one resource click selects the full %i-card bank rate",
  async (rate) => {
    const outgoing = Array(rate).fill("WOOD");
    while (outgoing.length < 4) outgoing.push(null);
    vi.mocked(getState).mockResolvedValue(
      makeGameState({
        player_state: { P1_WOOD_IN_HAND: 8 },
        current_playable_actions: [
          ["BLUE", "MARITIME_TRADE", [...outgoing, "ORE"]] as any,
        ],
      }),
    );
    show();
    await userEvent.click(await screen.findByRole("button", { name: "Trade" }));
    await userEvent.click(screen.getByRole("button", { name: "Lumber: 8" }));
    expect(
      screen.getByRole("button", { name: `Lumber: ${8 - rate}` }),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Ore: 19" }));
    expect(
      screen.getByRole("button", { name: "Confirm bank trade" }),
    ).toBeEnabled();
  },
);
function roadState() {
  return makeGameState({
    nodes: {
      ...makeGameState().nodes,
      "1": {
        id: 1,
        tile_coordinate: [0, 0, 0],
        direction: "NORTHWEST",
        color: null,
        building: null,
      },
    },
    current_playable_actions: [["BLUE", "BUILD_ROAD", [0, 1]]],
  });
}
test("build-button placement executes immediately, and toggling the button restores confirmation", async () => {
  vi.mocked(getState).mockResolvedValue(roadState());
  vi.mocked(postAction).mockResolvedValue(makeGameState({ state_index: 13 }));
  show();
  const user = userEvent.setup();
  await user.click(
    await screen.findByRole("button", { name: "Road" }),
  );
  await user.click(screen.getByRole("button", { name: "Road" }));
  await user.click(screen.getByRole("button", { name: "Build road 0 to 1" }));
  expect(
    screen.getByRole("button", { name: "Confirm road placement" }),
  ).toBeInTheDocument();
  expect(postAction).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Cancel placement" }));
  await user.click(screen.getByRole("button", { name: "Road" }));
  await user.click(screen.getByRole("button", { name: "Build road 0 to 1" }));
  expect(postAction).toHaveBeenCalledWith("game-123", [
    "BLUE",
    "BUILD_ROAD",
    [0, 1],
  ]);
});
test("Develop purchases immediately without a dialog", async () => {
  vi.mocked(getState).mockResolvedValue(
    makeGameState({
      current_playable_actions: [["BLUE", "BUY_DEVELOPMENT_CARD", null]],
    }),
  );
  vi.mocked(postAction).mockResolvedValue(makeGameState({ state_index: 13 }));
  show();
  await userEvent.click(
    await screen.findByRole("button", { name: "Develop" }),
  );
  expect(postAction).toHaveBeenCalledWith("game-123", [
    "BLUE",
    "BUY_DEVELOPMENT_CARD",
    null,
  ]);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});
test("fast-forward also rolls the dice", async () => {
  vi.mocked(getState).mockResolvedValue(makeGameState());
  vi.mocked(postAction).mockResolvedValue(
    makeGameState({
      state_index: 13,
      current_playable_actions: [["BLUE", "END_TURN", null]],
    }),
  );
  show();
  await userEvent.click(
    await screen.findByRole("button", { name: "Roll dice (fast forward)" }),
  );
  expect(postAction).toHaveBeenCalledWith("game-123", ["BLUE", "ROLL", null]);
});
