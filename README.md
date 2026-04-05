NCF — Neural Collaborative Filtering on MovieLens 1M

## Requirements

Python 3.8+
pip install torch numpy pandas matplotlib
No GPU needed. All three models train on CPU 

## Dataset
Download MovieLens 1M from https://grouplens.org/datasets/movielens/1m/
and place `ratings.dat` here:

ncf/
└── data/
    └── ml-1m/
        └── ratings.dat

The file uses `::` as a separator and has the format:
userId::movieId::rating::timestamp

## Project Structure

ncf/
├── data/ml-1m/ratings.dat
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── checkpoints/        
└── results/

## How to Run

Run all commands from the `ncf/` root folder, not from inside `src/`.

### Step 1 — Train GMF

Open `src/train.py` and set the CONFIG block at the top to:

```python
Config = {
    "model_type" : "gmf",
    "num_factors" : 32,
    "layers" : [64, 32, 16],
    "lr" : 0.001,
    "epochs" : 20,
    "patience" : 5,
    "checkpoint_path" : "checkpoints/gmf_32.pt",
    "history_path" : "results/train_gmf_32.json",
    "pretrain" : False,
}

Then run:
python src/train.py

GMF trains for roughly 10 epochs before early stopping. The best checkpoint
saves to `checkpoints/gmf_32.pt`.

### Step 2 — Train MLP
Change CONFIG in `src/train.py`:
"model_type" : "mlp",
"checkpoint_path" : "checkpoints/mlp_32.pt",
"history_path" : "results/train_mlp_32.json",
"pretrain" : False

Run:
python src/train.py

### Step 3 — Train NeuMF (with pre-training)
NeuMF loads weights from the GMF and MLP checkpoints saved in Steps 1 and 2.
Both must exist before running this step.

Change CONFIG in `src/train.py`:
"model_type" : "neumf",
"checkpoint_path" : "checkpoints/neumf_32.pt",
"history_path" : "results/train_neumf_32.json",
"pretrain" : True,
"gmf_checkpoint" : "checkpoints/gmf_32.pt",
"mlp_checkpoint" : "checkpoints/mlp_32.pt",
"pretrain_alpha" : 0.5,

Run:
python src/train.py

NeuMF initialises from the pre-trained GMF and MLP weights following
He et al. (2017) Section 3.4, then fine-tunes the full model.

### Step 4 — Evaluate all models
Once all three checkpoints exist, run:
python src/evaluate.py

This scores all three models using full-ranking evaluation, prints a results
table to the terminal, and saves two plots to `results/`:

`loss_curves.png` — training and validation BCE loss per model
`bar_charts.png` — recall@10 and ndcg@10 comparison across all models

Results are also saved to `results/eval_results.json`.

## Expected Results

Full-ranking evaluation against ~3,400 candidate items per user:

| Model              | Recall@10 | NDCG@10 |
|--------------------|-----------|---------|
| MLP  [64,32,16]    | ~0.129    | ~0.198  |
| GMF-only  emb=32   | ~0.146    | ~0.221  |
| NeuMF [64,32,16]   | ~0.147    | ~0.222  |

NeuMF > GMF > MLP on both metrics, matching the ordering in the original paper.