# Self-Pruning Neural Network (PyTorch)

This project demonstrates a self-pruning image classifier on CIFAR-10 using learnable gates inside linear layers.
The model learns both classification and sparsity jointly by adding an L1 penalty on gate activations.

## Overview

The core idea is to replace dense classifier layers with a custom `PrunableLinear` layer:

- Each weight has an associated learnable gate score.
- Gate scores are passed through a sigmoid to produce values in `(0, 1)`.
- Effective weights are computed as `weight * sigmoid(gate_scores)`.
- A sparsity term is added to training loss to push unimportant gates toward zero.

Total loss:

`total_loss = classification_loss + lambda * sparsity_loss`

Where `sparsity_loss` is the sum of gate values across all prunable layers.

## Model Architecture

The notebook defines:

- A CNN feature extractor (Conv-BN-ReLU blocks with pooling).
- A prunable classifier head:
	- `PrunableLinear(128 * 4 * 4, 256)`
	- `ReLU + Dropout`
	- `PrunableLinear(256, 10)`

## Repository Structure

```text
self-pruning-NN/
	pruning.ipynb      # Full implementation and experiment flow
	README.md          # Project documentation
	data/              # CIFAR-10 data directory (auto-created/used by torchvision)
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install torch torchvision matplotlib
```

3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open `pruning.ipynb` and run all cells.

## Experiment Flow in `pruning.ipynb`

1. Define `PrunableLinear`.
2. Build `PrunableNetwork`.
3. Define:
	 - `calculate_sparsity_loss(model)`
	 - `calculate_sparsity_level(model, threshold=1e-2)`
4. Train/evaluate via `train_and_evaluate(lambda_val, epochs, device)`.
5. Compare multiple lambda values (default in notebook: `0.0`, `1e-5`, `1e-4`).
6. Print final accuracy/sparsity summary.
7. Plot gate distribution for the best model.

## Notes

- The notebook automatically uses GPU when available:
	- `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- CIFAR-10 is downloaded with `torchvision.datasets.CIFAR10(..., download=True)`.
- Sparsity is measured as the percentage of gates below a threshold.

## Expected Outcome

By increasing `lambda`, you should generally observe:

- Higher sparsity in prunable layers.
- Potential trade-off in final classification accuracy.

This lets you explore the accuracy-sparsity frontier and choose a practical balance for compression.