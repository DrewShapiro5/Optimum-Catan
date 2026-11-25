$Params = @(
  '--n-cohorts', 16
  '--output', 'runs/catan-ai/cuda-cohorts-adversarial-v2-puct'
  '--cnn-checkpoint', 'runs/catan-ai/cuda-cohorts-adversarial-puct/cohort-010/cnn/iteration-1/best.pt'
  '--transformer-checkpoint', 'runs/catan-ai/cuda-cohorts-adversarial-puct/cohort-010/transformer/iteration-1/best.pt'
  '--mcts-opponent', 'adversarial-puct'
  '--simulations', 128
  '--setup-simulations', 512
  '--setup-exploration-games', 500
  '--setup-temperature', 1.5
  '--positions-per-game', 64
  '--resample-every', 100
  '--cnn-selfplay-games', 2000
  '--jsettlers-cnn-games', 0
  '--workers', 256
  '--coreml-batch', 128
  '--gpu-streams', 1
  '--wait-us', 50
  '--mcts-lanes', 1
  '--updates', 2000
  '--value-loss-weight', 0.25
  '--replay-cohorts', 4
  '--validation-fraction', 0.05
  '--cuda-fixed-batch'
  '--print-games'
  '--tail-over-dispatch', 32
  '--vp-margin-alpha', 0.1
)
.\run_cohort @Params