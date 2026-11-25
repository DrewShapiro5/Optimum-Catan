import axios from "axios";

import { API_URL } from "../configuration";
import type {
  Color,
  DebugSnapshot,
  GameAction,
  GameState,
} from "./api.types";

/** A registry key, e.g. "CATANATRON" or "AB". The server decides what exists. */
export type PlayerKey = string;

/** One settable parameter, as published by GET /api/players. */
export type PlayerParam = {
  name: string;
  type: "int" | "float" | "str" | "bool";
  default: number | string | boolean | null;
  /** Only present when the param declares a fixed set of values. */
  choices?: Array<string | number>;
  help: string;
};

export type PlayerEntry = {
  key: PlayerKey;
  name: string;
  description: string;
  is_bot: boolean;
  source: string;
  params: PlayerParam[];
};

export type MapTemplate = "BASE" | "MINI" | "TOURNAMENT";
export type StateIndex = number | `${number}` | "latest";

export type PlayerSpec =
  | PlayerKey
  | { key: PlayerKey; params?: Record<string, number | string | boolean> };

type CreateGameOptions = {
  players: PlayerSpec[];
  mapTemplate: MapTemplate;
  vpsToWin: number;
  discardLimit: number;
  friendlyRobber: boolean;
};

/** The players this server can seat. Replaces a hardcoded list in the UI. */
export async function getPlayers(): Promise<PlayerEntry[]> {
  const response = await axios.get<PlayerEntry[]>(API_URL + "/api/players");
  return response.data;
}

export async function createGame({
  players,
  mapTemplate,
  vpsToWin,
  discardLimit,
  friendlyRobber,
}: CreateGameOptions) {
  const response = await axios.post(API_URL + "/api/games", {
    players,
    map_template: mapTemplate,
    vps_to_win: vpsToWin,
    discard_limit: discardLimit,
    friendly_robber: friendlyRobber,
  });
  return response.data.game_id;
}

/**
 * Whose eyes the server renders state through. A seat sees the other hands as
 * counts rather than cards; null watches as a spectator, every hand face up.
 *
 * It lives here rather than as an argument because every request carries it,
 * and a call site that forgot would silently show the whole table again.
 */
let perspective: Color | null = null;

export function setPerspective(color: Color | null) {
  perspective = color;
}

function viewParams() {
  return perspective ? { as: perspective } : {};
}

export async function getState(
  gameId: string,
  stateIndex: StateIndex = "latest",
): Promise<GameState> {
  const response = await axios.get(
    `${API_URL}/api/games/${gameId}/states/${stateIndex}`,
    { params: viewParams() },
  );
  return response.data;
}

/** action=undefined means bot action */
export async function postAction(gameId: string, action?: GameAction) {
  const response = await axios.post<GameState>(
    `${API_URL}/api/games/${gameId}/actions`,
    action,
    { params: viewParams() },
  );
  return response.data;
}

export type MCTSProbabilities = {
  [K in Color]: number;
};

type MCTSSuccessBody = {
  success: true;
  probabilities: MCTSProbabilities;
  state_index: number;
};
type MCTSErrorBody = {
  success: false;
  error: string;
  trace: string;
};

export async function getMctsAnalysis(
  gameId: string,
  stateIndex: StateIndex = "latest",
) {
  try {
    console.log("Getting MCTS analysis for:", {
      gameId,
      stateIndex,
      url: `${API_URL}/api/games/${gameId}/states/${stateIndex}/mcts-analysis`,
    });

    if (!gameId) {
      throw new Error("No gameId provided to getMctsAnalysis");
    }

    const response = await axios.get<MCTSSuccessBody | MCTSErrorBody>(
      `${API_URL}/api/games/${gameId}/states/${stateIndex}/mcts-analysis`,
    );

    console.log("MCTS analysis response:", response.data);
    return response.data;
  } catch (error: any) {
    // AxiosResponse<MCTSErrorBody>
    console.error("MCTS analysis error:", {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data,
      stack: error.stack,
    });
    throw error;
  }
}

export type AgentEntry = {
  supported_rules?: { players?: number };
  searchable?: boolean;
  default_simulations?: number;
  default_mcts_opponent?: "self-policy" | "adversarial-puct" | "frozen-mlp";
  key: string;
  name: string;
  serverKey: string;
  available: boolean;
  description: string;
  reason: string;
  endpoint: string;
};
export async function getAgents(): Promise<AgentEntry[]> {
  return (await axios.get(API_URL + "/api/agents")).data.agents;
}

export async function getDebugState(gameId: string): Promise<DebugSnapshot> {
  const response = await axios.get<DebugSnapshot>(
    `${API_URL}/api/games/${gameId}/debug/state`,
  );
  return response.data;
}

export async function prepareDebugAction(
  gameId: string,
): Promise<DebugSnapshot> {
  const response = await axios.post<DebugSnapshot>(
    `${API_URL}/api/games/${gameId}/debug/prepare`,
  );
  return response.data;
}

export async function stepDebugAction(gameId: string): Promise<GameState> {
  const response = await axios.post<GameState>(
    `${API_URL}/api/games/${gameId}/debug/step`,
    {},
    { params: viewParams() },
  );
  return response.data;
}

export async function undoDebugAction(gameId: string): Promise<GameState> {
  const response = await axios.post<GameState>(
    `${API_URL}/api/games/${gameId}/debug/undo`,
    {},
    { params: viewParams() },
  );
  return response.data;
}

export async function setDebugSimulations(
  gameId: string,
  simulations: number,
): Promise<DebugSnapshot> {
  const response = await axios.post<DebugSnapshot>(
    `${API_URL}/api/games/${gameId}/debug/simulations`,
    { simulations },
  );
  return response.data;
}

export async function setDebugSearchSettings(
  gameId: string,
  settings: {
    simulations?: number;
    mcts_opponent?: "self-policy" | "adversarial-puct" | "frozen-mlp";
    vp_margin_alpha?: number;
  },
): Promise<DebugSnapshot> {
  const response = await axios.post<DebugSnapshot>(
    `${API_URL}/api/games/${gameId}/debug/simulations`,
    settings,
  );
  return response.data;
}
