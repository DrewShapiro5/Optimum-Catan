$Params = @(
  '--n-cohorts', 16
  '--output', 'runs/catan-ai/cuda-cohorts-adversarial-puct'
  '--mcts-opponent', 'adversarial-puct'
  '--simulations', 128
  '--setup-simulations', 512
  '--setup-exploration-games', 500
  '--setup-temperature', 1.5
  '--positions-per-game', 64
  '--resample-every', 100
  '--cnn-selfplay-games', 1000
  '--jsettlers-cnn-games', 0
  '--workers', 256
  '--coreml-batch', 128
  '--gpu-streams', 1
  '--wait-us', 50
  '--mcts-lanes', 1
  '--updates', 2000
  '--value-loss-weight', 0.25
  '--replay-cohorts', 2
  '--validation-fraction', 0.1
  '--cuda-fixed-batch'
  '--print-games'
)
.\run_cohort @Params