"""Predeclared, pair-clustered sequential futility checks for tuning only.

A pair score lies in [0,1]. Chernoff's Bernoulli-KL bound also applies to
independent bounded observations. Bonferroni across the scheduled looks makes
its one-sided coverage valid when stopping at any one of those looks. This is
an absolute performance screen, not a claim about an architecture or a seed.
"""
import math
from agent_apis.observation import require

DEFAULT={'version':1,'enabled':True,'minimum_pairs':20,'check_every_pairs':20,
         'alpha':0.05,'hopeless_match_score':0.25,
         'method':'bounded-pair-KL-upper-bound-with-finite-look-correction'}


def upper_bound(mean,n,alpha):
    require(n>0 and 0<=mean<=1 and 0<alpha<1,'Invalid futility bound inputs')
    if mean==1:return 1.
    if mean==0:return -math.expm1(math.log(alpha)/n)
    limit=math.log(1/alpha)/n;lo=mean;hi=1.
    for _ in range(80):
        q=(lo+hi)/2
        if q==1:break
        divergence=mean*math.log(mean/q)+(1-mean)*math.log((1-mean)/(1-q))
        if divergence>limit:hi=q
        else:lo=q
    return hi


def check(games,maximum_pairs,settings=None):
    policy=dict(DEFAULT if settings is None else settings)
    require(policy==DEFAULT or policy=={**DEFAULT,'enabled':False},'Unknown futility policy; version a new policy before changing thresholds')
    result={'policy':policy,'stopped':False,'scheduled_pairs':maximum_pairs}
    if not policy['enabled']:return result
    looks=list(range(policy['minimum_pairs'],maximum_pairs+1,policy['check_every_pairs']))
    if maximum_pairs>=policy['minimum_pairs'] and maximum_pairs not in looks:looks.append(maximum_pairs)
    grouped={}
    for game in games:grouped.setdefault(game['pair_id'],[]).append(game)
    require(all(len(g)==2 and {r['candidate_color'] for r in g}=={'BLUE','RED'} for g in grouped.values()),
            'Futility requires complete unique seat pairs')
    n=len(grouped);result['observed_pairs']=n
    if n not in looks:return result
    result['planned_looks']=looks
    # Errors/cutoffs require the existing failure gates, never a futility label.
    if any(g['status'] not in ('win','loss') for g in games):return result
    require(all(0<=g['match_score']<=1 for g in games),'Invalid match score')
    mean=sum(g['match_score'] for g in games)/len(games)
    bound=upper_bound(mean,n,policy['alpha']/len(looks))
    result.update(mean_match_score=mean,upper_match_score=bound,
                  confidence=1-policy['alpha'],alpha_per_look=policy['alpha']/len(looks),
                  stopped=bound<policy['hopeless_match_score'])
    if result['stopped']:result['reason']='Even the corrected one-sided upper bound is below 25% match score'
    return result
