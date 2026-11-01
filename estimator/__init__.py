"""The Stopping Estimator: a fine-tuned encoder that reads (question, retrieved path, candidate answer) and gives the
probability that the candidate answers the question. `render` writes its input, `server` serves a checkpoint,
`sampler` generates Data A from the graph and the gold, `train` holds the training notebook, the renaming and the
threshold."""
