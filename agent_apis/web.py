"""Provider-neutral discovery, decisions, and parallel-state synchronization."""

from flask import Blueprint, jsonify, request
from . import choose_action, list_agents, register_agents, synchronize, synchronize_game


def install(app):
    register_agents()
    bp = Blueprint("agent_apis", __name__, url_prefix="/api/agents")

    @bp.get("")
    def catalog():
        register_agents()
        return jsonify(
            {"protocol": "catanotron-observation-v1", "agents": list_agents()}
        )

    @bp.post("/<key>/decide")
    def decide(key):
        if request.content_length and request.content_length > 2_000_000:
            return jsonify(error="Observation exceeds 2 MB"), 413
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("observation"), dict):
            return jsonify(error="Expected an 'observation' object"), 400
        try:
            action = choose_action(key, body["observation"])
            return jsonify(action=action)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            return jsonify(error=str(error)), 400
        except (RuntimeError, OSError, TimeoutError) as error:
            return jsonify(error=str(error)), 503

    @bp.post("/<key>/sync")
    def sync(key):
        if request.content_length and request.content_length > 2_000_000:
            return jsonify(error="Observation exceeds 2 MB"), 413
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("observation"), dict):
            return jsonify(error="Expected an 'observation' object"), 400
        try:
            return jsonify(synchronize(key, body["observation"], body.get("color")))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            return jsonify(error=str(error)), 400
        except (RuntimeError, OSError, TimeoutError) as error:
            return jsonify(error=str(error)), 503

    @app.after_request
    def update_seated_providers(response):
        if response.status_code != 200 or request.endpoint not in (
            "api.post_game_endpoint",
            "api.post_action_endpoint",
            "agent_debug.debug_step",
            "agent_debug.debug_undo",
        ):
            return response
        payload = response.get_json(silent=True) or {}
        gid = (request.view_args or {}).get("game_id") or payload.get("game_id")
        if gid:
            from catanatron.web.models import get_game_state

            try:
                synchronize_game(get_game_state(gid))
            except (ValueError, RuntimeError, OSError, TimeoutError):
                # The engine has already committed the move. Don't turn that
                # success into a failed action that a client might submit twice.
                # The next decision rehydrates from the authoritative snapshot.
                app.logger.exception("Agent state synchronization failed for %s", gid)
                response.headers["X-Agent-Sync"] = "failed"
        return response

    debug_bp = Blueprint("agent_debug", __name__, url_prefix="/api/games")

    @debug_bp.get("/<string:game_id>/debug/state")
    def debug_state(game_id):
        from . import debug_telemetry

        # Fast-path: return active in-memory telemetry without hitting SQLite on
        # every 120 ms poll while a search or step-pause is in progress.
        snap = debug_telemetry.get_snapshot(game_id)
        if snap is not None and snap.get("status") in ("searching", "awaiting_step"):
            return jsonify(snap)

        from catanatron.web.models import get_game_state

        game = get_game_state(game_id)
        if game is None:
            return jsonify(error="Game not found"), 404
        state_index = len(game.state.action_records)
        if snap is None:
            snap = debug_telemetry.snapshot_from_game(game, status="idle")
        elif game.state.current_player().is_bot and snap.get("state_index") != state_index:
            snap = debug_telemetry.snapshot_from_game(game, status="idle")
        return jsonify(snap)

    @debug_bp.post("/<string:game_id>/debug/prepare")
    def debug_prepare(game_id):
        from catanatron.serialization import action_to_json
        from catanatron.web.models import get_game_state
        from . import debug_telemetry

        with debug_telemetry.game_lock(game_id):
            game = get_game_state(game_id)
            if game is None:
                return jsonify(error="Game not found"), 404
            if game.winning_color() is not None or not game.state.current_player().is_bot:
                return jsonify(debug_telemetry.snapshot_from_game(game, status="idle"))
            state_index = len(game.state.action_records)
            existing = debug_telemetry.get_pending_action(game_id, state_index)
            if existing is not None:
                snap = debug_telemetry.get_snapshot(game_id)
                if snap is not None and snap.get("state_index") == state_index:
                    return jsonify(snap)
            player = game.state.current_player()
            with debug_telemetry.track_decision(game, hold=True) as outcome:
                action = player.decide(game, game.playable_actions)
                outcome["action_json"] = action_to_json(action)
            return jsonify(debug_telemetry.get_snapshot(game_id))

    @debug_bp.post("/<string:game_id>/debug/step")
    def debug_step(game_id):
        import json as _json
        from flask import Response
        from catanatron.serialization import action_from_json, action_to_json
        from catanatron.web.api import _browser_view
        from catanatron.web.models import get_game_state, upsert_game_state
        from . import debug_telemetry

        with debug_telemetry.game_lock(game_id):
            game = get_game_state(game_id)
            if game is None:
                return jsonify(error="Game not found"), 404
            if game.winning_color() is not None or not game.state.current_player().is_bot:
                debug_telemetry.clear_pending_action(game_id)
                return Response(
                    response=_json.dumps(_browser_view(game)),
                    status=200,
                    mimetype="application/json",
                )
            state_index = len(game.state.action_records)
            pending = debug_telemetry.get_pending_action(game_id, state_index)
            if pending is not None:
                action = action_from_json(pending)
                for observer in game._steppers:
                    observer.step(game, action)
                game.execute(action)
                upsert_game_state(game)
                debug_telemetry.clear_pending_action(game_id)
                snap = debug_telemetry.get_snapshot(game_id)
                if snap is not None and snap.get("state_index") == state_index:
                    with debug_telemetry._LOCK:
                        snap["status"] = "completed"
                        debug_telemetry._SNAPSHOTS[game_id] = snap
            else:
                with debug_telemetry.track_decision(game, hold=False) as outcome:
                    record = game.play_tick()
                    outcome["action_json"] = action_to_json(record.action)
                upsert_game_state(game)
            return Response(
                response=_json.dumps(_browser_view(game)),
                status=200,
                mimetype="application/json",
            )

    @debug_bp.post("/<string:game_id>/debug/undo")
    def debug_undo(game_id):
        import json as _json
        from flask import Response
        from catanatron.serialization import state_to_json
        from catanatron.web.api import _browser_view
        from catanatron.web.models import StoredAction, StoredGame, db, get_game_state
        from . import debug_telemetry

        with debug_telemetry.game_lock(game_id):
            stored = db.session.get(StoredGame, game_id)
            if stored is None:
                return jsonify(error="Game not found"), 404
            if stored.head_index > stored.base_index:
                prev_index = stored.head_index - 1
                prev_game = get_game_state(game_id, prev_index)
                db.session.query(StoredAction).filter(
                    StoredAction.uuid == game_id,
                    StoredAction.index >= prev_index,
                ).delete()
                stored = db.session.get(StoredGame, game_id)
                stored.head_index = prev_index
                stored.head_state = _json.dumps(state_to_json(prev_game))
                db.session.commit()
            else:
                prev_game = get_game_state(game_id)
            debug_telemetry.clear_pending_action(game_id)
            with debug_telemetry._LOCK:
                debug_telemetry._SNAPSHOTS[game_id] = debug_telemetry.snapshot_from_game(
                    prev_game, status="idle"
                )
            return Response(
                response=_json.dumps(_browser_view(prev_game)),
                status=200,
                mimetype="application/json",
            )

    @debug_bp.post("/<string:game_id>/debug/simulations")
    def debug_simulations(game_id):
        from dataclasses import replace as dc_replace
        from catanatron.web.models import get_game_state
        from . import debug_telemetry

        body = request.get_json(silent=True) or {}
        has_sims = "simulations" in body and body["simulations"] is not None
        has_opp = "mcts_opponent" in body and body["mcts_opponent"] is not None
        has_alpha = "vp_margin_alpha" in body and body["vp_margin_alpha"] is not None
        if not has_sims and not has_opp and not has_alpha:
            return jsonify(error="Provide 'simulations', 'mcts_opponent', and/or 'vp_margin_alpha'"), 400

        sims = None
        if has_sims:
            try:
                sims = int(body["simulations"])
            except (TypeError, ValueError):
                return jsonify(error="'simulations' must be an integer between 0 and 4096"), 400
            if not 0 <= sims <= 4096:
                return jsonify(error="'simulations' must be between 0 and 4096"), 400

        mcts_opp = None
        if has_opp:
            mcts_opp = str(body["mcts_opponent"])
            if mcts_opp not in ("self-policy", "adversarial-puct", "frozen-mlp"):
                return jsonify(error="'mcts_opponent' must be 'self-policy', 'adversarial-puct', or 'frozen-mlp'"), 400

        vp_alpha = None
        if has_alpha:
            try:
                vp_alpha = float(body["vp_margin_alpha"])
            except (TypeError, ValueError):
                return jsonify(error="'vp_margin_alpha' must be a number between 0.0 and 0.5"), 400
            if not (0.0 <= vp_alpha <= 0.5):
                return jsonify(error="'vp_margin_alpha' must be between 0.0 and 0.5"), 400

        with debug_telemetry.game_lock(game_id):
            game = get_game_state(game_id)
            if game is None:
                return jsonify(error="Game not found"), 404
            updated = 0
            for player in game.state.players:
                params = getattr(player, "params", None)
                if not (player.is_bot and params is not None):
                    continue
                patch = {}
                if sims is not None and hasattr(params, "simulations"):
                    patch["simulations"] = sims
                if mcts_opp is not None and hasattr(params, "mcts_opponent"):
                    patch["mcts_opponent"] = mcts_opp
                if vp_alpha is not None and hasattr(params, "vp_margin_alpha"):
                    patch["vp_margin_alpha"] = vp_alpha
                if patch:
                    player.params = dc_replace(params, **patch)
                    updated += 1
            if updated:
                import json as _json
                from catanatron.registry import REGISTRY
                from catanatron.web.models import StoredGame, db
                stored = db.session.get(StoredGame, game_id)
                if stored is not None:
                    stored.player_specs = _json.dumps(
                        [REGISTRY.spec_of(p) for p in game.state.players]
                    )
                    db.session.commit()
            debug_telemetry.clear_pending_action(game_id)
            cur_sims, can_sims = debug_telemetry.game_simulations_info(game)
            cur_opp, can_opp = debug_telemetry.game_mcts_opponent_info(game)
            cur_alpha, can_alpha = debug_telemetry.game_vp_margin_alpha_info(game)
            snap = debug_telemetry.get_snapshot(game_id)
            if snap is not None:
                with debug_telemetry._LOCK:
                    snap["simulations"] = cur_sims
                    snap["simulations_target"] = cur_sims
                    snap["can_set_simulations"] = can_sims
                    snap["mcts_opponent"] = cur_opp
                    snap["can_set_mcts_opponent"] = can_opp
                    snap["vp_margin_alpha"] = cur_alpha
                    snap["can_set_vp_margin_alpha"] = can_alpha
                    if snap.get("status") == "awaiting_step":
                        snap["status"] = "idle"
                    debug_telemetry._SNAPSHOTS[game_id] = snap
            else:
                snap = debug_telemetry.snapshot_from_game(game, status="idle")
            return jsonify(snap)

    app.register_blueprint(bp)
    app.register_blueprint(debug_bp)
