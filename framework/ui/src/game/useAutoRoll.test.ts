import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { useAutoRoll } from "./useAutoRoll";
beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});
test("automatically rolls once at three seconds", () => {
  const roll = vi.fn();
  renderHook(() => useAutoRoll("turn-1", false, roll));
  act(() => vi.advanceTimersByTime(2999));
  expect(roll).not.toHaveBeenCalled();
  act(() => vi.advanceTimersByTime(1));
  expect(roll).toHaveBeenCalledOnce();
  act(() => vi.advanceTimersByTime(3000));
  expect(roll).toHaveBeenCalledOnce();
});
test("manual rolling removes the timer and the next turn gets three new seconds", () => {
  const roll = vi.fn();
  const { rerender } = renderHook(
    ({ turn }) => useAutoRoll(turn, false, roll),
    { initialProps: { turn: "turn-1" as string | null } },
  );
  act(() => vi.advanceTimersByTime(1000));
  rerender({ turn: null });
  act(() => vi.advanceTimersByTime(3000));
  expect(roll).not.toHaveBeenCalled();
  rerender({ turn: "turn-2" });
  act(() => vi.advanceTimersByTime(2999));
  expect(roll).not.toHaveBeenCalled();
  act(() => vi.advanceTimersByTime(1));
  expect(roll).toHaveBeenCalledOnce();
});
test("UI updates don't reset the deadline, and unmount cancels it", () => {
  const roll = vi.fn();
  const { rerender, unmount } = renderHook(
    ({ callback }) => useAutoRoll("turn-1", false, callback),
    { initialProps: { callback: roll } },
  );
  act(() => vi.advanceTimersByTime(2000));
  const latest = vi.fn();
  rerender({ callback: latest });
  act(() => vi.advanceTimersByTime(1000));
  expect(roll).not.toHaveBeenCalled();
  expect(latest).toHaveBeenCalledOnce();
  unmount();
  act(() => vi.advanceTimersByTime(4000));
  expect(latest).toHaveBeenCalledOnce();
});
test("an action in progress delays sending until it completes without another three-second wait", () => {
  const roll = vi.fn();
  const { rerender } = renderHook(
    ({ blocked }) => useAutoRoll("turn-1", blocked, roll),
    { initialProps: { blocked: true } },
  );
  act(() => vi.advanceTimersByTime(4000));
  expect(roll).not.toHaveBeenCalled();
  rerender({ blocked: false });
  act(() => vi.advanceTimersByTime(1));
  expect(roll).toHaveBeenCalledOnce();
});
