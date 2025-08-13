# Ultimate Tic‑Tac‑Toe AI

> A collection of agents and training experiments for **Ultimate Tic‑Tac‑Toe**, plus a Jupyter notebook for exploration.

## Overview
This repo contains several agent implementations and experimental scripts:
- `brute_force_agent.py` — search/heuristic baseline.
- `svm_agent.py` — scikit‑learn SVM policy/value approach.
- `nn_agent*.py`, `nn_version*.py` — multiple PyTorch prototypes for learned evaluation/policy.
- `agent.py` — common interfaces or utilities for agents (see code).
- `mini-project.ipynb` — notebook with experiments and comparisons.
- `utils.py` — game utilities (board state, moves, helpers).
- `model/` — saved model weights/checkpoints.
- `figures/` — plots or visualizations from experiments.
- `data.pkl` — sample training/evaluation data (serialized).

> Note: `data.pkl` and `model/` can be large. Consider using **Git LFS** if you plan to store trained weights or big datasets.

## Install
Requires **Python ≥ 3.10**.
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart
### Notebook (recommended)
```bash
jupyter lab   # or: jupyter notebook
```
Open `mini-project.ipynb` and run all cells to reproduce baseline comparisons or train toy models.

### Script experiments
This codebase is organized by standalone scripts. Typical usage patterns:
- Edit an agent (e.g., `brute_force_agent.py`, `svm_agent.py`, or a `nn_*.py` file).
- Run your test script (e.g., `test.py`) to pit two agents against each other and print win/draw/loss statistics.
- Save figures to `figures/` and model weights to `model/`.

Because scripts may evolve, see each file’s `__main__` guard or inline comments for the exact CLI flags.

## Repo Structure
```
.
├── agent.py
├── brute_force_agent.py
├── svm_agent.py
├── nn_agent.py, nn_agent1.py, nn_agent_hardcode.py, nn_version2.py, ...
├── training1.py, test.py, tmp.py
├── utils.py
├── model/                 # saved weights/checkpoints
├── figures/               # training/eval plots
├── data.pkl               # sample data (optional, large)
├── mini-project.ipynb
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Development Tips
- **Reproducibility**: set random seeds for Python/NumPy/PyTorch; log results across N games as both Player 1 and Player 2.
- **Large files**: avoid committing big binaries; prefer **Git LFS** or publishing links. Add patterns to `.gitignore`.
- **Results**: keep a simple `RESULTS.md` (win rates vs. baselines) and include a small figure or table in the README.
- **Packaging**: if you plan to grow the project, create a package (e.g., `utt/`) with `agents/`, `env/`, and a `cli.py` runner.

## Citation
If you build on this code, please credit “SUN WEIYANG — Ultimate Tic‑Tac‑Toe AI (GitHub)” and include a link to the repository.

## License
This project is released under the MIT License (see `LICENSE`).