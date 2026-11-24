import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { createGame, getAgents } from "../utils/apiClient";
import HomePage from "./HomePage";
vi.mock("../utils/apiClient", () => ({
  createGame: vi.fn(),
  getAgents: vi.fn(),
}));
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getAgents).mockResolvedValue([]);
  localStorage.clear();
});
afterEach(cleanup);
function renderMenu() {
  render(
    <MemoryRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/games/:id" element={<div>Game opened</div>} />
      </Routes>
    </MemoryRouter>,
  );
}
test("defaults to two human seats and does not offer a nonexistent API", () => {
  renderMenu();
  expect(screen.getByRole("button", { name: /2 players/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(screen.getAllByRole("combobox")).toHaveLength(2);
  for (const select of screen.getAllByRole("combobox")) {
    expect(select).toHaveValue("human");
  }
  expect(
    screen.getAllByRole("option", { name: "API · none connected" })[0],
  ).toBeDisabled();
  expect(
    screen.queryByRole("button", { name: /3 players/ }),
  ).not.toBeInTheDocument();
});
test("four seats create exactly four humans and preserve player names", async () => {
  vi.mocked(createGame).mockResolvedValue("table-four");
  const user = userEvent.setup();
  renderMenu();
  await user.click(screen.getByRole("button", { name: /4 players/ }));
  expect(screen.getAllByRole("combobox")).toHaveLength(4);
  await user.clear(screen.getByLabelText("Player 1 name"));
  await user.type(screen.getByLabelText("Player 1 name"), "Drew");
  await user.click(screen.getByRole("button", { name: /^Start game$/ }));
  expect(await screen.findByText("Game opened")).toBeInTheDocument();
  expect(createGame).toHaveBeenCalledWith(
    expect.objectContaining({
      players: ["HUMAN", "HUMAN", "HUMAN", "HUMAN"],
      vpsToWin: 10,
      discardLimit: 7,
      friendlyRobber: true,
    }),
  );
  expect(
    JSON.parse(localStorage.getItem("catanatron.players.table-four")!),
  ).toMatchObject({ RED: "Drew", WHITE: "Player 4" });
});
test("switching back to two seats sends only two and reports server failure", async () => {
  vi.mocked(createGame).mockRejectedValue(new Error("offline"));
  const user = userEvent.setup();
  renderMenu();
  await user.click(screen.getByRole("button", { name: /4 players/ }));
  await user.click(screen.getByRole("button", { name: /2 players/ }));
  await user.click(screen.getByRole("button", { name: /^Start game$/ }));
  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent("Couldn't start"),
  );
  expect(createGame).toHaveBeenCalledWith(
    expect.objectContaining({
      players: ["HUMAN", "HUMAN"],
      vpsToWin: 15,
      discardLimit: 9,
      friendlyRobber: true,
    }),
  );
  expect(screen.getByRole("button", { name: /^Start game$/ })).toBeEnabled();
});

test("loads API players from the central catalog and sends their server key", async () => {
  vi.mocked(getAgents).mockResolvedValue([
    {
      key: "JSETTLERS",
      serverKey: "JSETTLERS",
      name: "JSettlers (estimator adapter)",
      available: true,
      description: "Native estimator",
      reason: "",
      endpoint: "/api/agents/JSETTLERS/decide",
    },
  ]);
  vi.mocked(createGame).mockResolvedValue("bot-game");
  renderMenu();
  const user = userEvent.setup();
  await screen.findAllByRole("option", { name: "JSettlers (estimator adapter)" });
  await user.selectOptions(screen.getByLabelText("Player 2 type"), "jsettlers");
  await user.click(screen.getByRole("button", { name: "Start game" }));
  expect(createGame).toHaveBeenCalledWith(
    expect.objectContaining({ players: ["HUMAN", "JSETTLERS"] }),
  );
});

test("CNN inspection agents are selectable only for two players", async () => {
  vi.mocked(getAgents).mockResolvedValue([{
    key: "CNN_INSPECT_SEARCH", serverKey: "CNN_INSPECT_SEARCH",
    name: "CNN best measured — search (500 ms)", available: true,
    description: "Pinned XD-CNN", reason: "", endpoint: "/api/agents/CNN_INSPECT_SEARCH/decide",
    supported_rules: { players: 2 },
  }]);
  renderMenu();
  const user = userEvent.setup();
  await screen.findAllByRole("option", { name: /CNN best measured/ });
  await user.selectOptions(screen.getByLabelText("Player 2 type"), "cnn_inspect_search");
  expect(screen.getByLabelText("Player 2 type")).toHaveValue("cnn_inspect_search");
  await user.click(screen.getByRole("button", { name: /4 players/ }));
  expect(screen.getByLabelText("Player 2 type")).toHaveValue("human");
  for (const option of screen.getAllByRole("option", { name: /CNN best measured/ })) {
    expect(option).toBeDisabled();
  }
});

test("searchable models show a tunable simulations input and send it to createGame", async () => {
  vi.mocked(getAgents).mockResolvedValue([
    {
      key: "CNN_LATEST_SEARCH",
      serverKey: "CNN_LATEST_SEARCH",
      name: "CNN latest best — search",
      available: true,
      searchable: true,
      default_simulations: 128,
      description: "Latest XD-CNN",
      reason: "",
      endpoint: "/api/agents/CNN_LATEST_SEARCH/decide",
      supported_rules: { players: 2 },
    },
  ]);
  vi.mocked(createGame).mockResolvedValue("search-game");
  renderMenu();
  const user = userEvent.setup();
  await screen.findAllByRole("option", { name: /CNN latest best/ });
  expect(screen.queryByLabelText("Player 2 simulations")).not.toBeInTheDocument();
  await user.selectOptions(screen.getByLabelText("Player 2 type"), "cnn_latest_search");
  const simsInput = screen.getByLabelText("Player 2 simulations");
  expect(simsInput).toHaveValue(128);
  await user.clear(simsInput);
  await user.type(simsInput, "256");
  await user.click(screen.getByRole("button", { name: "Start game" }));
  expect(createGame).toHaveBeenCalledWith(
    expect.objectContaining({
      players: [
        "HUMAN",
        { key: "CNN_LATEST_SEARCH", params: { simulations: 256 } },
      ],
    }),
  );
});
