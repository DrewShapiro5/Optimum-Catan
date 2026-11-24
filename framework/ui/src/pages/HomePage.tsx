import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createGame, getAgents, type AgentEntry } from "../utils/apiClient";
import { COLORS, COLOR_HEX, HUMAN, saveNames } from "../game/model";
import Icon from "../game/Icons";
import "../game/colonist.scss";
export default function HomePage() {
  const [agents, setAgents] = useState<AgentEntry[]>([]);
  const [agentError, setAgentError] = useState("");
  useEffect(() => {
    let active = true;
    getAgents()
      .then((entries) => {
        if (active) setAgents(entries);
      })
      .catch(() => {
        if (active)
          setAgentError("Agent APIs could not be loaded. Check the server.");
      });
    return () => {
      active = false;
    };
  }, []);
  const API_PLAYERS = agents.map((a) => ({ ...a, key: a.key.toLowerCase() }));
  const [count, setCount] = useState<2 | 4>(2);
  const [names, setNames] = useState([
    "Player 1",
    "Player 2",
    "Player 3",
    "Player 4",
  ]);
  const [providers, setProviders] = useState([
    "human",
    "human",
    "human",
    "human",
  ]);
  const [simulations, setSimulations] = useState([128, 128, 128, 128]);
  const [mctsOpponents, setMctsOpponents] = useState<string[]>([
    "self-policy",
    "self-policy",
    "self-policy",
    "self-policy",
  ]);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  const navigate = useNavigate();
  async function start() {
    if (loading) return;
    setLoading(true);
    setError("");
    try {
      const all = [HUMAN, ...API_PLAYERS];
      const id = await createGame({
        players: providers.slice(0, count).map((p, i) => {
          const found = all.find((a) => a.key === p)!;
          const apiAgent = API_PLAYERS.find((a) => a.key === p);
          if (apiAgent?.searchable) {
            const sims = Math.max(
              0,
              Math.min(4096, Math.round(simulations[i] || 0)),
            );
            const params: Record<string, number | string | boolean> = {
              simulations: sims,
            };
            if (apiAgent.default_mcts_opponent) {
              params.mcts_opponent = mctsOpponents[i] || apiAgent.default_mcts_opponent;
            }
            return { key: found.serverKey, params };
          }
          return found.serverKey;
        }),
        mapTemplate: "BASE",
        vpsToWin: count === 2 ? 15 : 10,
        discardLimit: count === 2 ? 9 : 7,
        friendlyRobber: true,
      });
      saveNames(id, names.slice(0, count));
      navigate(`/games/${id}`);
    } catch {
      setError(
        "Couldn't start the game. Check that the local server is running and try again.",
      );
      setLoading(false);
    }
  }
  return (
    <div className="settlers-app lobby">
      <header className="lobby-header">
        <a href="/" className="brand">
          <span className="brand-mark">
            <Icon name="settlement" />
          </span>
          CATANATRON<span className="brand-tag">LOCAL TABLE</span>
        </a>
        <span className="connection">
          <i /> Local game
        </span>
      </header>
      <div className="lobby-content">
        <section
          className="setup-panel cream-panel"
          aria-labelledby="setup-title"
        >
          <div className="setup-heading">
            <div>
              <h2 id="setup-title">Start a game</h2>
            </div>
            <Icon name="settlement" />
          </div>
          <label className="field-caption">Number of players</label>
          <div
            className="player-count"
            role="group"
            aria-label="Number of players"
          >
            {([2, 4] as const).map((n) => (
              <button
                key={n}
                onClick={() => {
                  setCount(n);
                  setProviders(providers.map(p => {
                    const agent = API_PLAYERS.find(a => a.key === p);
                    return agent?.supported_rules?.players && agent.supported_rules.players !== n ? "human" : p;
                  }));
                }}
                aria-pressed={count === n}
                className={count === n ? "selected" : ""}
              >
                <Icon name={n === 2 ? "person" : "army"} />
                <span>
                  {n} players
                  <small>
                    {n === 2 ? "15 victory points" : "10 victory points"}
                  </small>
                </span>
                {count === n && <span className="selected-check">✓</span>}
              </button>
            ))}
          </div>
          <div className="seats-heading">
            <span className="field-caption">Players</span>
            <span>{count} seats · shared screen</span>
          </div>
          <div className="seat-list">
            {COLORS.slice(0, count).map((color, i) => {
              const selectedAgent = API_PLAYERS.find(
                (a) => a.key === providers[i],
              );
              return (
                <div className="seat-row" key={color}>
                  <span
                    className="avatar"
                    style={
                      {
                        "--player-color": COLOR_HEX[color],
                      } as React.CSSProperties
                    }
                  >
                    <Icon name="person" />
                  </span>
                  <label className="seat-name">
                    <span>{color.toLowerCase()} seat</span>
                    <input
                      aria-label={`Player ${i + 1} name`}
                      maxLength={24}
                      value={names[i]}
                      onChange={(e) =>
                        setNames(
                          names.map((n, j) => (j === i ? e.target.value : n)),
                        )
                      }
                    />
                  </label>
                  <select
                    aria-label={`Player ${i + 1} type`}
                    value={providers[i]}
                    onChange={(e) => {
                      const nextKey = e.target.value;
                      setProviders(
                        providers.map((p, j) => (j === i ? nextKey : p)),
                      );
                      const nextAgent = API_PLAYERS.find(
                        (a) => a.key === nextKey,
                      );
                      if (nextAgent?.searchable) {
                        const def = nextAgent.default_simulations ?? 128;
                        setSimulations(
                          simulations.map((s, j) => (j === i ? def : s)),
                        );
                        if (nextAgent.default_mcts_opponent) {
                          const defOpp = nextAgent.default_mcts_opponent;
                          setMctsOpponents(
                            mctsOpponents.map((m, j) => (j === i ? defOpp : m)),
                          );
                        }
                      }
                    }}
                  >
                    <option value="human">Human</option>
                    {API_PLAYERS.map((p) => (
                      <option
                        key={p.key}
                        value={p.key}
                        disabled={!p.available || !!(p.supported_rules?.players && p.supported_rules.players !== count)}
                        title={p.reason || p.description}
                      >
                        {p.name}
                      </option>
                    ))}
                    {API_PLAYERS.length === 0 && (
                      <option disabled>API · none connected</option>
                    )}
                  </select>
                  {selectedAgent?.searchable && (
                    <label className="seat-sims">
                      <span>MCTS simulations</span>
                      <input
                        type="number"
                        aria-label={`Player ${i + 1} simulations`}
                        min={0}
                        max={4096}
                        step={16}
                        value={simulations[i]}
                        onChange={(e) => {
                          const parsed = Number(e.target.value);
                          setSimulations(
                            simulations.map((s, j) =>
                              j === i ? (Number.isFinite(parsed) ? parsed : 0) : s,
                            ),
                          );
                        }}
                      />
                      <small>0 = policy only</small>
                    </label>
                  )}
                  {selectedAgent?.default_mcts_opponent && (
                    <label className="seat-sims">
                      <span>MCTS opponent</span>
                      <select
                        aria-label={`Player ${i + 1} MCTS opponent`}
                        value={mctsOpponents[i]}
                        onChange={(e) => {
                          const val = e.target.value;
                          setMctsOpponents(
                            mctsOpponents.map((m, j) => (j === i ? val : m)),
                          );
                        }}
                      >
                        <option value="self-policy">Self-policy</option>
                        <option value="adversarial-puct">Adversarial PUCT (no-stop)</option>
                        <option value="frozen-mlp">Frozen MLP</option>
                      </select>
                    </label>
                  )}
                </div>
              );
            })}
          </div>
          <p className="api-note">
            <span className="api-dot" />
            {agentError ||
              (agents.length
                ? `${agents.filter(a => a.available).length} agent API available.`
                : "No API players connected yet. All seats are human.")}
          </p>
          <div className="setup-rules">
            <span>Classic island</span>
            <span>{count === 2 ? 15 : 10} victory points</span>
            <span>Discard above {count === 2 ? 9 : 7}</span>
          </div>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <button
            className="start-game"
            aria-label="Start game"
            onClick={start}
            disabled={loading}
          >
            <Icon name="settlement" />
            {loading ? "Starting…" : "Start game"}
            <span>→</span>
          </button>
          <p className="setup-footnote">Shared screen · Open hands</p>
        </section>
      </div>
      <footer className="lobby-footer">
        <span>Built on the Catanatron game engine</span>
      </footer>
    </div>
  );
}
