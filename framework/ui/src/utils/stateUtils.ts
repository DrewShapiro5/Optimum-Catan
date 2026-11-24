import type { Color, GameState } from "./api.types";

/**
 * Check if it's a human player's turn
 * @param gameState
 * @returns True if a human player needs to play
 */
export function isPlayersTurn(gameState: GameState): boolean {
  return !gameState.bot_colors.includes(gameState.current_color);
}

export function playerKey(gameState: GameState, color: Color): string {
  return `P${gameState.colors.indexOf(color)}`;
}

export function getHumanColor(gameState: GameState): Color {
  // In shared-screen games, controls and cards follow whoever must act now.
  if (!gameState.bot_colors.includes(gameState.current_color)) {
    return gameState.current_color;
  }
  return gameState.colors.find(
    (color) => !gameState.bot_colors.includes(color)
  ) as Color;
}
