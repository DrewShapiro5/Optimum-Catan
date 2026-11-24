"""Recover outcome margins and action patterns by verified benchmark replay."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
from catanatron.models.player import Color
from catanatron.gym.envs.action_space import from_action_space
from catanatron.serialization import action_record_to_json
from catanatron.state_functions import get_actual_victory_points,player_key
from agent_apis.benchmark import COLORS,make_game
from agent_apis.experiment import ROOT,digest,file_digest
from agent_apis.observation import require


def replay(folder,neural_sample_games=20):
    results=[json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
    events=defaultdict(list)
    for line in (folder/'timings.jsonl').read_text().splitlines():
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        events[event['pair_id'],event['candidate_color']].append(event)
    recovered=[]
    config=json.loads((folder/'configuration.json').read_text())
    from agent_apis.neural.provider import NeuralCheckpoint
    model=NeuralCheckpoint(ROOT/config['candidate']['checkpoint']['path'],device='cpu') if neural_sample_games else None
    for game_index,original in enumerate(results):
        game=make_game(original);candidate=Color[original['candidate_color']]
        counts={c:Counter() for c in COLORS};opportunities={c:Counter() for c in COLORS}
        opening={};predictions=[];teacher_checks=[];teacher_nonforced=0
        for event in events[original['pair_id'],original['candidate_color']]:
            require(event['action_id'] is not None,'Incomplete action inside a completed game')
            actor=game.state.current_color();require(actor.value==event['actor'],'Replay actor differs')
            action=from_action_space(event['action_id'],actor,COLORS,'BASE')
            require(action in game.playable_actions,'Recorded action is illegal on replay')
            kinds={a.action_type.value for a in game.playable_actions}
            counts[actor][action.action_type.value]+=1
            if model and game_index<neural_sample_games:
                own_end=actor==candidate and action.action_type.value=='END_TURN' and counts[actor]['END_TURN'] in (1,10,20,40)
                if actor!=candidate and len(game.playable_actions)>1:teacher_nonforced+=1
                check_teacher=actor!=candidate and len(game.playable_actions)>1 and teacher_nonforced%10==0
                if own_end or check_teacher:
                    from catanatron.serialization import web_view
                    from agent_apis.observation import from_web_view
                    view=from_web_view(web_view(game,actor),actor.value)
                    probabilities,value=model.evaluate(view)
                    if own_end:predictions.append({'own_end_turn':counts[actor]['END_TURN'],'win_probability':value,
                        'actual_vp':get_actual_victory_points(game.state,candidate),'opponent_vp':get_actual_victory_points(game.state,next(c for c in COLORS if c!=candidate))})
                    if check_teacher:teacher_checks.append({'action_type':action.action_type.value,
                        'correct':max(probabilities,key=lambda i:(probabilities[i],-i))==event['action_id'],
                        'target_probability':probabilities[event['action_id']]})
            if action.action_type.value=='END_TURN':
                opportunities[actor]['end_turns']+=1
                for name,options in [('settlement_or_city',{'BUILD_SETTLEMENT','BUILD_CITY'}),
                                     ('any_build',{'BUILD_SETTLEMENT','BUILD_CITY','BUILD_ROAD'}),
                                     ('development_card',{'BUY_DEVELOPMENT_CARD'})]:
                    opportunities[actor]['end_with_'+name+'_available']+=int(bool(kinds&options))
            was_initial=game.state.is_initial_build_phase
            game.execute(action)
            if was_initial and not game.state.is_initial_build_phase:
                for color in COLORS:
                    production=Counter()
                    for node in game.state.buildings_by_color[color]['SETTLEMENT']:production.update(game.state.board.map.node_production[node])
                    opening[color.value]=dict(production)
        require(digest([action_record_to_json(r) for r in game.state.action_records])==original['trajectory_sha256'],
                'Replay trajectory hash differs; recovered scores are not valid')
        require(len(game.state.action_records)==original['actions'],'Replay action count differs')
        require((game.winning_color().value if game.winning_color() else None)==original['winner'],'Replay winner differs')
        sides={}
        for color in COLORS:
            key=player_key(game.state,color);ps=game.state.player_state;build=game.state.buildings_by_color[color]
            sides['candidate' if color==candidate else 'opponent']={
                'vp':get_actual_victory_points(game.state,color),'actions':dict(counts[color]),
                'end_turn_opportunities':dict(opportunities[color]),
                'settlements':len(build['SETTLEMENT']),'cities':len(build['CITY']),'roads':len(build['ROAD']),
                'resource_cards':sum(ps[f'{key}_{r}_IN_HAND'] for r in ('WOOD','BRICK','SHEEP','WHEAT','ORE')),
                'longest_road_length':ps[f'{key}_LONGEST_ROAD_LENGTH']}
        recovered.append({**original,'opening_production':opening,'candidate_value_samples':predictions,'frozen_opponent_checks':teacher_checks,'terminal':sides,'vp_margin':sides['candidate']['vp']-sides['opponent']['vp']})
    return recovered


def summarize(rows):
    summary={'games':len(rows),'wins':sum(r['status']=='win' for r in rows),'losses':sum(r['status']=='loss' for r in rows),
             'errors':sum(r['status']=='error' for r in rows),'cutoffs':sum(r['status']=='cutoff' for r in rows),
             'vp_margin_mean':statistics.mean(r['vp_margin'] for r in rows),
             'vp_margin_median':statistics.median(r['vp_margin'] for r in rows),
             'candidate_vp_histogram':dict(sorted(Counter(r['terminal']['candidate']['vp'] for r in rows).items())),
             'mean_turns':statistics.mean(r['turns'] for r in rows),'seats':{}}
    for role in ('candidate','opponent'):
        actions=Counter();opportunities=Counter()
        for r in rows:actions.update(r['terminal'][role]['actions']);opportunities.update(r['terminal'][role]['end_turn_opportunities'])
        summary[role]={'mean_'+k:statistics.mean(r['terminal'][role][k] for r in rows)
                       for k in ('vp','settlements','cities','roads','resource_cards','longest_road_length')}
        summary[role]['actions_per_game']={k:v/len(rows) for k,v in actions.items()}
        summary[role]['end_turn_opportunities']=dict(opportunities)
    values=[sample for r in rows for sample in r['candidate_value_samples']]
    summary['own_play_value_samples']={str(turn):{'count':len(selected),
        'mean_predicted_win_probability':statistics.mean(s['win_probability'] for s in selected),
        'mean_actual_vp':statistics.mean(s['actual_vp'] for s in selected),
        'mean_opponent_vp':statistics.mean(s['opponent_vp'] for s in selected)}
        for turn in (1,10,20,40) if (selected:=[s for s in values if s['own_end_turn']==turn])}
    teacher=[sample for r in rows for sample in r['frozen_opponent_checks']]
    summary['frozen_opponent_agreement']={}
    for label,subset in [('nonforced',teacher),('nonforced_excluding_roll',[s for s in teacher if s['action_type']!='ROLL'])]:
        if subset:summary['frozen_opponent_agreement'][label]={'samples':len(subset),
             'top1_accuracy':statistics.mean(s['correct'] for s in subset),
             'mean_target_probability':statistics.mean(s['target_probability'] for s in subset)}
    for role in ('candidate','opponent'):
        resources=[]
        for r in rows:
            c=r['candidate_color'] if role=='candidate' else ('RED' if r['candidate_color']=='BLUE' else 'BLUE')
            resources.append(r['opening_production'][c])
        summary[role]['opening_cards_per_roll']=statistics.mean(sum(p.values()) for p in resources)
        summary[role]['opening_resource_production']={k:statistics.mean(p.get(k,0) for p in resources) for k in ('WOOD','BRICK','SHEEP','WHEAT','ORE')}
        summary[role]['opening_missing_wood_or_brick_games']=sum(not p.get('WOOD') or not p.get('BRICK') for p in resources)
    for seat in (0,1):
        subset=[r for r in rows if r['candidate_start_seat']==seat]
        summary['seats'][str(seat)]={'games':len(subset),'mean_candidate_vp':statistics.mean(r['terminal']['candidate']['vp'] for r in subset),
                                    'mean_vp_margin':statistics.mean(r['vp_margin'] for r in subset)}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folders',nargs='+',type=Path);args=parser.parse_args()
    report={'purpose':'Read-only, exact trajectory-verified replay; no new evaluation games','runs':{}}
    for folder in args.folders:
        rows=replay(folder);config=json.loads((folder/'configuration.json').read_text())
        key=config['candidate']['key'];entry={'folder':str(folder),'games_sha256':file_digest(folder/'games.jsonl'),
             'timings_sha256':file_digest(folder/'timings.jsonl'),'configuration':config,'trajectory_hashes_verified':len(rows),
             'summary':summarize(rows),'games':rows}
        report['runs'][key]=entry;print(key,json.dumps(entry['summary'],indent=2),flush=True)
    path=ROOT/'runs/catan-ai/phase5-loss-diagnosis.json';path.write_text(json.dumps(report,indent=2)+'\n');print(path)


if __name__=='__main__':main()
