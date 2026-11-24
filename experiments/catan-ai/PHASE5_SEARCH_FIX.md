# Long-history search recursion fix

The CNN's original 500 ms search screening recorded 52 wins, 147 losses and one
error in 200 games. The error was `RecursionError` at tuning-0074, candidate RED,
turn 905/action 3156. It remains a failed run and is not eligible for promotion.
The record is preserved at
`runs/catan-ai/phase3-NEURAL_SEARCH-5e3c8fe1a55f-20261105T050207455838Z/run.json`.

The permitted observation was reconstructed from logged seeds/actions and its
trajectory hash verified. Development-card belief contained 492 events. The
recursive cached probability traversal also used generator/caller frames, so
this legitimate history was close to Python's recursion limit; adding 20 caller
frames reproduced the exception directly.

The probability graph is acyclic. It now uses an explicit DFS stack and cache
while retaining transitions, summation order, event order and RNG consumption.
The exact failing posterior evidence and 20 seeded sampled worlds match the
original implementation bit for bit. The reconstructed neural decision now
returns a legal action in a two-simulation regression. Additional tests cover
more than 1,000 belief events and compare recursive/iterative probabilities and
100 seeded worlds on a shorter conditioned history. The existing cache-size
instrumentation API is preserved.

Source hashes and validation evidence are in [phase5-belief-fix.json](phase5-belief-fix.json).
Because execution cost can change the number of simulations completed in 500 ms,
CNN search is rerun under a distinct resolved configuration. Failed and corrected
runs are never pooled. Pure policies never evaluate development beliefs, so their
completed runs are retained with compatibility provenance. The zero-win MLP search
baseline remains an explicitly historical pre-fix control, never a promotion
candidate. No training checkpoint, native SMART code, time budget or frozen seed
list changed.
