import { test, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";
vi.mock("./utils/apiClient", () => ({getAgents: vi.fn().mockResolvedValue([])}));
test("opens the two-player starting menu", () => {
  render(<App />);
  expect(
    screen.getByRole("heading", { name: "Start a game" }),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /2 players/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(screen.getByRole("button", { name: /4 players/ })).toBeInTheDocument();
  expect(screen.getAllByRole("combobox")).toHaveLength(2);
  expect(screen.getByRole("button", { name: /^Start game$/ })).toBeEnabled();
});
