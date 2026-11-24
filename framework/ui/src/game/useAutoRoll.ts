import { useEffect, useRef, useState } from "react";

/** One deadline per human turn. UI updates and pre-roll card plays do not reset it. */
export function useAutoRoll(
  turn: string | null,
  blocked: boolean,
  onRoll: () => void,
) {
  const callback = useRef(onRoll);
  callback.current = onRoll;
  const deadline = useRef<{ turn: string; at: number } | null>(null);
  const [seconds, setSeconds] = useState<number | null>(null);
  useEffect(() => {
    if (!turn) {
      deadline.current = null;
      setSeconds(null);
      return;
    }
    if (deadline.current?.turn !== turn)
      deadline.current = { turn, at: Date.now() + 3000 };
    const at = deadline.current.at;
    const update = () =>
      setSeconds(Math.max(0, Math.ceil((at - Date.now()) / 1000)));
    update();
    if (blocked) return;
    const interval = window.setInterval(update, 200);
    const timer = window.setTimeout(
      () => callback.current(),
      Math.max(0, at - Date.now()),
    );
    return () => {
      window.clearInterval(interval);
      window.clearTimeout(timer);
    };
  }, [turn, blocked]);
  return seconds;
}
