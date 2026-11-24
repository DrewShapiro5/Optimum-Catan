# Tuning evaluation futility amendment v1

Added at the user's request after the original phase-5 MLP evaluation. Historical
runs and the frozen benchmark protocol/seed files remain unchanged. This is a
prospective amendment: future resolved configurations record the policy under
`evaluation.futility`. It applies to tuning, not to reserved final confirmation.

The default stops a hopeless run after a complete board/seat pair when a
corrected one-sided 95% upper bound is below **25% match score**. Check after
20 pairs and every 20 pairs thereafter, including the scheduled endpoint. The
maximum remains the original tier size: smoke 20, screening 100, expanded 500.
There are no new boards, seat-dependent stops, rerolls or substituted games.

Treat the mean of the two seat scores as one independent bounded observation in
[0,1]. For n pairs and L scheduled looks, compute the Bernoulli-KL Chernoff upper
bound U satisfying `n * KL(observed_mean || U) = log(L / 0.05)`. Chernoff's bound
also holds for bounded observations; the finite-look correction controls the
one-sided error over the scheduled looks. Both seats are always completed
before inspecting the rule. The confidence statement is per evaluation run,
not across every model in the experiment and not a superiority claim.

With zero wins and no cutoffs after 20 pairs (40 games), the bound is 13.9% in a
smoke run, or 20.6% at the first look of a 100-pair screening run. Both are below
25%, so extension is unnecessary. These positive bounds avoid the misleading
[0,0] percentile bootstrap interval obtained by resampling only observed losses.

The implementation is `agent_apis/futility.py`. Tests cover the analytic
zero-win bound, correction for repeated looks, incomplete/duplicate seat-pair
rejection, competitive runs continuing, exclusion of errors/cutoffs from a
futility label, actual harness stopping, continuation rejection and prevention
of downstream pipeline jobs.

A stopped run has status `stopped-early`; logs, checkpoints, completed pair IDs,
scheduled/actual counts, reason, policy version, corrected upper bound and
source/configuration provenance are preserved. Its ordinary fixed-sample
bootstrap interval is suppressed; a descriptive bootstrap is retained under a
separate label. It cannot be promoted or extended as a completed full run.
The original neural pipeline halts for diagnosis after futility. Following the
user’s subsequent request to finish evaluation, `neural.evaluation` continues
with other model/mode comparisons while preserving each futile result. It never
extends that futile run, and additional seeds are limited to architectures with
a feasible, non-futile mode. Search is tested even when policy fails. Historical
MLP losses are imported with explicit provenance, not relabeled as prospective
early stops. An all-futile comparison produces no competitive winner. For search
promotion across eligible finalists, the paired three-seed comparison uses the
prospectively fixed first 20 pairs per seed, collected before any stopping
decision; variable-length stopped samples are otherwise descriptive.

Errors and illegal actions remain failures, not losses eligible for this rule.
Cutoffs retain the existing cutoff gate and are not classified as futility.
Every future game also records both players' terminal public/actual victory
points and the candidate's margin, so severity can be inspected immediately.
