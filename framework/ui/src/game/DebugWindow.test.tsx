import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { StateProvider } from "../store";
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
} from "../utils/apiClient";
import { makeGameState } from "../test/fixtures";
import type { DebugSnapshot } from "../utils/api.types";
import GameScreen from "../pages/GameScreen";

vi.mock("../utils/apiClient", () => ({
  getState: vi.fn(),
  postAction: vi.fn(),
  setPerspective: vi.fn(),
  getDebugState: vi.fn(),
  prepareDebugAction: vi.fn(),
  stepDebugAction: vi.fn(),
  undoDebugAction: vi.fn(),
  setDebugSimulations: vi.fn(),
  setDebugSearchSettings: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

afterEach(cleanup);

function renderGame() {
  render(
    <StateProvider>
      <MemoryRouter initialEntries={["/games/game-123"]}>
        <Routes>
          <Route
            path="/games/:gameId"
            element={<GameScreen replayMode={false} />}
          />
        </Routes>
      </MemoryRouter>
    </StateProvider>,
  );
}

function makeDebugSnapshot(
  overrides: Partial<DebugSnapshot> = {},
): DebugSnapshot {
  return {
    game_id: "game-123",
    state_index: 12,
    color: "RED",
    is_bot: true,
    player_label: "CNN latest best — search",
    status: "awaiting_step",
    mode: "search",
    simulations_done: 64,
    simulations_target: 128,
    simulations: 128,
    can_set_simulations: true,
    root_value: 0.612,
    updated_at: 1700000000,
    dev_cards: {
      KNIGHT: 2,
      MONOPOLY: 1,
      YEAR_OF_PLENTY: 0,
      ROAD_BUILDING: 0,
      VICTORY_POINT: 1,
    },
    resources: { WOOD: 4, BRICK: 1, SHEEP: 0, WHEAT: 2, ORE: 3 },
    actions: [
      {
        action: ["RED", "BUILD_SETTLEMENT", 0],
        action_id: 10,
        policy: 0.45,
        prior: 0.35,
        visits: 29,
        mean_value: 0.64,
        selected: true,
      },
      {
        action: ["RED", "BUILD_ROAD", [0, 1]],
        action_id: 11,
        policy: 0.2,
        prior: 0.25,
        visits: 13,
        mean_value: 0.58,
        selected: false,
      },
      {
        action: ["RED", "BUILD_CITY", 1],
        action_id: 12,
        policy: 0.1,
        prior: 0.1,
        visits: 6,
        mean_value: 0.55,
        selected: false,
      },
      {
        action: ["RED", "MOVE_ROBBER", [[0, 0, 0], "BLUE"]],
        action_id: 13,
        policy: 0.1,
        prior: 0.12,
        visits: 6,
        mean_value: 0.52,
        selected: false,
      },
      {
        action: [
          "RED",
          "MARITIME_TRADE",
          ["WOOD", "WOOD", "WOOD", "WOOD", "ORE"],
        ],
        action_id: 14,
        policy: 0.15,
        prior: 0.18,
        visits: 10,
        mean_value: 0.59,
        selected: false,
      },
    ],
    chosen_action: ["RED", "BUILD_SETTLEMENT", 0],
    ...overrides,
  };
}

test("expands debug window, displays dev hand and legal trades, dims board, and shows hover tooltip", async () => {
  const game = makeGameState({
    current_color: "RED",
    bot_colors: ["RED"],
    state_index: 12,
    nodes: {
      "0": {
        id: 0,
        tile_coordinate: [0, 0, 0],
        direction: "NORTH",
        color: null,
        building: null,
      },
      "1": {
        id: 1,
        tile_coordinate: [0, 0, 0],
        direction: "NORTHWEST",
        color: "RED",
        building: "SETTLEMENT",
      },
    },
  });
  const snap = makeDebugSnapshot({ status: "searching" });
  vi.mocked(getState).mockResolvedValue(game);
  vi.mocked(postAction).mockReturnValue(new Promise(() => {}));
  vi.mocked(getDebugState).mockResolvedValue(snap);

  renderGame();
  const toggle = await screen.findByRole("button", {
    name: "Expand debug window",
  });
  await userEvent.click(toggle);

  expect(await screen.findByTestId("debug-window")).toBeInTheDocument();
  expect(screen.getByText("Development Card Hand")).toBeInTheDocument();
  expect(screen.queryByText("Legal Trades")).not.toBeInTheDocument();
  expect(await screen.findByText("Trade 4 Lumber → 1 Ore")).toBeInTheDocument();

  // Board base layer should be dimmed and policy overlay layer should appear.
  const baseLayer = screen.getByTestId("board-base-layer");
  expect(baseLayer).toHaveClass("dimmed");
  expect(screen.getByTestId("policy-overlay-layer")).toBeInTheDocument();

  // Hovering a board policy overlay shows the assigned probability tooltip.
  const settleOverlay = screen.getByLabelText("Build Settlement · #0: 45.0%");
  fireEvent.mouseEnter(settleOverlay);
  const tooltip = await screen.findByTestId("policy-tooltip");
  expect(tooltip).toHaveTextContent("Build Settlement · #0");
  expect(tooltip).toHaveTextContent("45.0%");
  expect(tooltip).toHaveTextContent("29 visits");
});

test("wait between actions pauses bot moves and enables Take action button", async () => {
  const humanState = makeGameState({
    current_color: "BLUE",
    bot_colors: ["RED"],
    state_index: 11,
  });
  const botState = makeGameState({
    current_color: "RED",
    bot_colors: ["RED"],
    state_index: 12,
  });
  const nextHumanState = makeGameState({
    current_color: "BLUE",
    bot_colors: ["RED"],
    state_index: 13,
  });
  const snap = makeDebugSnapshot({ state_index: 12, status: "awaiting_step" });

  let resolvePrepare!: (s: DebugSnapshot) => void;
  const preparePromise = new Promise<DebugSnapshot>((resolve) => {
    resolvePrepare = resolve;
  });
  vi.mocked(getState).mockResolvedValue(humanState);
  vi.mocked(postAction).mockResolvedValue(botState);
  vi.mocked(getDebugState).mockResolvedValue(snap);
  vi.mocked(prepareDebugAction).mockReturnValueOnce(preparePromise).mockResolvedValue(snap);
  vi.mocked(stepDebugAction).mockResolvedValue(nextHumanState);
  vi.mocked(setDebugSimulations).mockResolvedValue({
    ...snap,
    simulations: 256,
    simulations_target: 256,
  });

  renderGame();
  await userEvent.click(
    await screen.findByRole("button", { name: "Expand debug window" }),
  );
  const waitCheckbox = screen.getByLabelText("Wait between actions");
  const stepBtn = screen.getByRole("button", { name: "Take action →" });
  expect(stepBtn).toBeDisabled();

  await userEvent.click(waitCheckbox);
  expect(waitCheckbox).toBeChecked();

  // Human rolls dice, handing turn to RED bot.
  await userEvent.click(screen.getByRole("button", { name: "Roll dice" }));

  // Because Wait between actions is ON, prepareDebugAction is called instead of auto-stepping.
  await waitFor(() =>
    expect(prepareDebugAction).toHaveBeenCalledWith("game-123"),
  );
  expect(postAction).toHaveBeenCalledTimes(1); // Only the human's ROLL
  expect(stepBtn).toBeDisabled();

  // Resolve the async MCTS preparation — Take action button must become enabled.
  resolvePrepare(snap);
  await waitFor(() => expect(stepBtn).toBeEnabled());

  // Updating search simulations calls setDebugSimulations.
  const simInput = screen.getByLabelText("Search simulations");
  await userEvent.clear(simInput);
  await userEvent.type(simInput, "256");
  await userEvent.click(screen.getByRole("button", { name: "Set" }));
  await waitFor(() =>
    expect(setDebugSimulations).toHaveBeenCalledWith("game-123", 256),
  );

  await waitFor(() => expect(stepBtn).toBeEnabled());
  await userEvent.click(stepBtn);

  await waitFor(() =>
    expect(stepDebugAction).toHaveBeenCalledWith("game-123"),
  );
  expect(setPerspective).toHaveBeenCalledWith(null);
});

test("vp-margin alpha control displays current value and updates via setDebugSearchSettings", async () => {
  const game = makeGameState({
    current_color: "RED",
    bot_colors: ["RED"],
    state_index: 12,
  });
  const snap = makeDebugSnapshot({
    can_set_simulations: true,
    can_set_vp_margin_alpha: true,
    vp_margin_alpha: 0.03,
  });
  vi.mocked(getState).mockResolvedValue(game);
  vi.mocked(postAction).mockReturnValue(new Promise(() => {}));
  vi.mocked(getDebugState).mockResolvedValue(snap);
  vi.mocked(setDebugSearchSettings).mockResolvedValue({
    ...snap,
    vp_margin_alpha: 0.15,
  });

  renderGame();
  await userEvent.click(
    await screen.findByRole("button", { name: "Expand debug window" }),
  );

  const alphaInput = await screen.findByLabelText("VP-margin alpha");
  expect(alphaInput).toHaveValue(0.03);

  await userEvent.clear(alphaInput);
  await userEvent.type(alphaInput, "0.15");
  await userEvent.click(
    screen.getByRole("button", { name: "Set VP-margin alpha" }),
  );

  await waitFor(() =>
    expect(setDebugSearchSettings).toHaveBeenCalledWith("game-123", {
      vp_margin_alpha: 0.15,
    }),
  );
  expect(undoDebugAction).not.toHaveBeenCalled();
});
