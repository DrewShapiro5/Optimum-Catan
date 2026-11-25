"""Re-root compatible search subtrees using permitted observations only.

Real history must extend the prior request and start with its selected action.
Hidden-card beliefs are always rebuilt. If opponent purchases have ever made
worlds uncertain, visits/returns are reset while compatible states and priors
remain reusable. Different sessions/models/settings never share a tree.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
from catanatron.serialization import action_from_json
from agent_apis.benchmark import named_seed
from agent_apis.development_belief import CARDS
from agent_apis.experiment import digest
from agent_apis.observation import require,validate_rules
from agent_apis.stochastic import action_id,observation_key,policy_observation
from agent_apis.stochastic_search import StochasticSearch
from agent_apis.neural.parallel_search import BatchedEvaluator


def match_key(view):
    # Includes static board, exact public/private snapshot, full permitted
    # normalized history and legal actions, beyond the model feature vector.
    return digest(policy_observation(view))


def historical_action_id(payload):
    payload=list(payload)
    # Serialized purchase records include the observed drawn type; that type
    # is a chance outcome, not part of the 332-action BUY action identity.
    if payload[1] in ('BUY_DEVELOPMENT_CARD','ROLL'):payload[2]=None
    return action_id(action_from_json(payload))


def allocation_key(game,root):
    other=next(c for c in game.state.colors if c.value!=root)
    seat=game.state.colors.index(other);prefix=f'P{seat}_';counts=Counter();draws={}
    for step,record in enumerate(game.state.action_records):
        if record.action.color!=other:continue
        kind=record.action.action_type.value
        if kind=='END_TURN':counts.clear()
        elif kind=='BUY_DEVELOPMENT_CARD':
            counts[record.result]+=1;draws[step]=record.result
    new=tuple(counts[c] for c in CARDS)
    old=tuple(game.state.player_state[prefix+c+'_IN_HAND']-new[i] for i,c in enumerate(CARDS))
    require(min(old)>=0,'Invalid cached development-card ages')
    deck=Counter(game.state.development_listdeck)
    # VP has no engine playability flag; its start flag is belief-derived.
    played_since=Counter()
    for record in reversed(game.state.action_records):
        if record.action.color!=other:continue
        if record.action.action_type.value=='END_TURN':break
        if record.action.action_type.value.startswith('PLAY_'):
            from agent_apis.observation import PLAY_TYPES
            card=PLAY_TYPES.get(record.action.action_type.value)
            if card:played_since[card]+=1
    flags=tuple(bool(old[i] or played_since[c]) for i,c in enumerate(CARDS))
    return old,new,tuple(draws.items()),tuple(deck[c] for c in CARDS),flags


class ReusableSearch(StochasticSearch):
    def __init__(self,*args,session_id=None,max_retained_nodes=4096,max_retained_states=8192,**kwargs):
        super().__init__(*args,**kwargs)
        self.session_id=session_id;self.max_retained_nodes=max_retained_nodes
        self.max_retained_states=max_retained_states
        self.positions={};self.selected_action=None;self.retained_root=None
        self.metrics.update(reused_nodes=0,reused_root_visits=0,reused_world_states=0,
                            reuse_statistics_reset=False,reuse_reset_reason='fresh',reuse_seconds=0.,
                            successor_cache_hits=0,chance_cache_hits=0)

    def _step(self,game,edge):
        before=self.metrics['retained_chance_children'];classes=self.metrics['chance_distributions']
        result=super()._step(game,edge)
        self.metrics['successor_cache_hits']+=int(self.metrics['retained_chance_children']==before)
        self.metrics['chance_cache_hits']+=int(self.metrics['chance_distributions']==classes)
        return result

    def _simulate(self,game,path,root_depth,action_depth):
        if path not in self.positions and game.state.current_color().value==self.root and game.winning_color() is None:
            # The search already needs this cached feature/knowledge key. Full
            # history comparisons are delayed until the next real decision.
            self.positions[path]=(self._view_key(game,self.root),game)
        return super()._simulate(game,path,root_depth,action_depth)

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        self.selected_action=action_id(action_from_json(result['action']))
        return result

    def advance(self,view,seed,evaluator,opponent,session_id):
        start=time.perf_counter()
        fresh=type(self)(view,seed,evaluator,opponent,merged=self.merged,c_puct=self.c_puct,
             max_root_depth=self.max_root_depth,max_actions=self.max_actions,
             mcts_opponent=self.mcts_opponent,vp_margin_alpha=self.vp_margin_alpha,
             session_id=session_id,max_retained_nodes=self.max_retained_nodes,max_retained_states=self.max_retained_states)
        def reset(reason):
            fresh.nodes={};fresh.opponent_nodes={};fresh.opponent_edges={};fresh.positions={};fresh.world_templates={}
            fresh.decoded_world=None;fresh.view_cache={};fresh.view_keys={};fresh.world_keys={};fresh.retained_root=None
            fresh.metrics.update(reuse_reset_reason=reason,reuse_seconds=time.perf_counter()-start)
            return fresh
        if session_id is None or session_id!=self.session_id:return reset('different-or-missing-session')
        if fresh.root!=self.root:return reset('different-perspective')
        old_history=self.view['action_records'];new_history=fresh.view['action_records'];delta=len(new_history)-len(old_history)
        if delta<=0 or new_history[:len(old_history)]!=old_history:return reset('history-rewind-or-divergence')
        if self.selected_action is None or historical_action_id(new_history[len(old_history)][0])!=self.selected_action:
            return reset('different-played-action')
        key=match_key(fresh.view)
        observation_id=observation_key(fresh.view)
        candidates=[(p,g) for p,(k,g) in self.positions.items() if len(p)==delta and k==observation_id and
                    match_key(self._view(g,self.root))==key]
        matches=[p for p,g in candidates if p in self.nodes]
        if not matches:return reset('unexplored-observation')
        prefix=max(matches,key=lambda p:(self.nodes[p].visits,repr(p)))
        retained={p[len(prefix):]:node for p,node in self.nodes.items() if p[:len(prefix)]==prefix}
        retained_opp={p[len(prefix):]:node for p,node in self.opponent_nodes.items() if p[:len(prefix)]==prefix}
        if len(retained)+len(retained_opp)>self.max_retained_nodes:return reset('retained-node-limit')
        fresh.nodes=retained
        fresh.opponent_nodes=retained_opp
        fresh.opponent_edges={(p[len(prefix):],a):edge for (p,a),edge in self.opponent_edges.items() if p[:len(prefix)]==prefix}
        fresh.positions={p[len(prefix):]:v for p,v in self.positions.items() if p[:len(prefix)]==prefix}
        # A new posterior is sampled in run(), never inherited from old worlds.
        # Returning statistics would otherwise mix different hidden-world beliefs
        # and strategic action-likelihood conditioning from the old search.
        uncertain=any(a[0]!=self.root and a[1]=='BUY_DEVELOPMENT_CARD' for a,_ in new_history)
        reset_stats=uncertain or self.max_root_depth is not None or self.metrics['capped_simulations']>0
        if reset_stats:
            seen=set()
            edges=list(fresh.opponent_edges.values())
            for node in list(fresh.nodes.values())+list(fresh.opponent_nodes.values()):node.visits=0;edges.extend(node.edges.values())
            for edge in edges:
                if id(edge) in seen:continue
                seen.add(id(edge));edge.visits=0;edge.avail=0;edge.total=0.
                for world in edge.worlds.values():
                    for child in world.children.values():child.visits=0;child.total=0.
        root_games=[g for p,g in candidates]
        for game in root_games[:128]:fresh.world_templates[allocation_key(game,self.root)]=game
        fresh.decoded_world=root_games[0]
        live=set(root_games)
        for _,game in fresh.positions.values():live.add(game)
        for node in list(fresh.nodes.values())+list(fresh.opponent_nodes.values()):
            for edge in node.edges.values():
                for world in edge.worlds.values():live.update(child.successor for child in world.children.values())
        for edge in fresh.opponent_edges.values():
            for world in edge.worlds.values():live.update(child.successor for child in world.children.values())
        if len(live)>self.max_retained_states:return reset('retained-state-limit')
        fresh.view_cache={k:v for k,v in self.view_cache.items() if k[0] in live}
        fresh.view_keys={k:v for k,v in self.view_keys.items() if k[0] in live}
        fresh.world_keys={k:v for k,v in self.world_keys.items() if k in live}
        fresh.retained_root=fresh.nodes[()]
        require(fresh.retained_root.observation==observation_key(fresh.view),'Re-root alias changed permitted observation')
        fresh.metrics.update(reused_nodes=len(fresh.nodes),reused_root_visits=fresh.retained_root.visits,
            reused_world_states=len(live),reuse_statistics_reset=reset_stats,
            reuse_reset_reason='hidden-belief-or-horizon-reset' if reset_stats else 'compatible-subtree',
            reuse_seconds=time.perf_counter()-start)
        return fresh


class SearchSession:
    """Persistent per-runtime, per-seat worker trees; inference services are renewed."""
    def __init__(self):self.trees=[];self.signature=None;self.lock=threading.RLock()
    def clear(self):
        with self.lock:self.trees=[];self.signature=None
    def run(self,view,seed,evaluator,opponent,workers,simulations,context,c_puct=1.5,max_actions=128,session_id=None,model_revision=None,adversarial_puct=False,mcts_opponent=None,vp_margin_alpha=0.03):
        require(type(workers)is int and 1<=workers<=8 and simulations>0,'Invalid search budget')
        validate_rules(view);workers=min(workers,simulations)
        if mcts_opponent is None:
            mcts_opponent='adversarial-puct' if adversarial_puct else 'frozen-mlp'
        with self.lock:
            signature=(session_id,view['perspective'],id(evaluator),id(opponent),workers,c_puct,max_actions,model_revision,mcts_opponent,float(vp_margin_alpha))
            previous=self.trees if session_id is not None and signature==self.signature else []
            start=time.perf_counter();policy=BatchedEvaluator(evaluator,workers);enemy=BatchedEvaluator(opponent,workers)
            from agent_apis import debug_telemetry
            parent_tracker=debug_telemetry.current_tracker()
            try:
                def run(i):
                    with debug_telemetry.bind_tracker(parent_tracker):
                        worker_context=context
                        if context is not None and context.mode=='time' and workers>1:
                            worker_context=SimpleNamespace(mode='time',deadline=context.deadline,
                                admission_deadline=min(context.admission_deadline,context.deadline-.050))
                        worker_seed=seed if workers==1 else named_seed(seed,f'root-worker-{i}')
                        tree=(previous[i].advance(view,worker_seed,policy,enemy,session_id) if previous else
                              ReusableSearch(view,worker_seed,policy,enemy,c_puct=c_puct,max_actions=max_actions,mcts_opponent=mcts_opponent,vp_margin_alpha=vp_margin_alpha,session_id=session_id))
                        if not previous and self.trees:tree.metrics['reuse_reset_reason']='configuration-or-session-changed'
                        result=tree.run(simulations//workers+int(i<simulations%workers),worker_context)
                        return tree,result
                with ThreadPoolExecutor(max_workers=workers) as pool:pairs=list(pool.map(run,range(workers)))
            except BaseException:
                self.trees=[];self.signature=None;raise
            finally:policy.close();enemy.close()
            self.trees=[t for t,r in pairs];self.signature=signature
            combined={}
            for _,result in pairs:
                for row in result['root_statistics']:
                    item=combined.setdefault(row['action_id'],{'action_id':row['action_id'],'visits':0,'total':0.,'prior':row['prior']})
                    item['visits']+=row['visits'];item['total']+=row['visits']*row['mean_value']
            if not combined:return pairs[0][1]
            rows=[{**r,'mean_value':r['total']/r['visits'] if r['visits'] else .5} for r in combined.values()]
            selected=max(rows,key=lambda r:(r['visits'],r['mean_value'],-r['action_id']))
            action=next(a for a in view['current_playable_actions'] if action_id(action_from_json(a))==selected['action_id'])
            # Each independent worker must check the action actually played by
            # the aggregated root, rather than its own local favorite.
            for tree in self.trees:tree.selected_action=selected['action_id']
            metrics={'workers':workers,'elapsed_seconds':time.perf_counter()-start,'worker_metrics':[r['metrics'] for _,r in pairs],
                     'inference_batches':policy.batches+enemy.batches,'inference_requests':policy.requests+enemy.requests,
                     'inference_service_seconds':policy.forward_seconds+enemy.forward_seconds,'subtree_reuse_enabled':True}
            for name in ('simulations','reused_nodes','reused_root_visits','reused_world_states','reuse_seconds','root_world_cache_hits','root_world_decodes','successor_cache_hits','chance_cache_hits'):
                metrics[name]=sum(r['metrics'].get(name,0) for _,r in pairs)
            for name in ('max_root_depth','max_action_depth','max_completed_turn_depth'):
                metrics[name]=max(r['metrics'][name] for _,r in pairs)
            metrics['statistics_reset_workers']=sum(r['metrics']['reuse_statistics_reset'] for _,r in pairs)
            metrics['reuse_reset_reasons']=dict(Counter(r['metrics']['reuse_reset_reason'] for _,r in pairs))
            metrics['simulations_per_second']=metrics['simulations']/metrics['elapsed_seconds']
            return {'action':action,'root_statistics':rows,'metrics':metrics}
