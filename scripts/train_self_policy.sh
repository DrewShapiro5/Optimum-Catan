./run_coreml_cohort.sh \
  --n-cohorts 8 \
  --cnn-checkpoint runs/catan-ai/jsettlers-expert-cohorts-visible/cohort-011/iteration-1/best.pt \
  --transformer-checkpoint runs/catan-ai/sparse-transformer-expert-catchup-visible/cohort-011/iteration-1/best.pt \
  --output runs/catan-ai/coreml-cohorts-v2-selfpolicy \
  --adversarial-puct \
  --cnn-selfplay-games 1500 \
  --jsettlers-cnn-games 500 \
  --replay-cohorts 3 \
  --workers 256 --mcts-lanes 1 --coreml-batch 32 --gpu-streams 8 --wait-us 100
