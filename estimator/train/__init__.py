"""Training the Stopping Estimator. `stop_judge.ipynb` is the training notebook as it was run (data, audit,
cleaning, rendering, model selection, fine-tuning, threshold, DAgger rounds, transfer); `rename` builds the released
estimator's training rows (renamed copies); `threshold` picks tau from scores. Needs the `[estimator]` extra and, to
fine-tune, a GPU."""
