# DSLR

42-style data science project: explore a Hogwarts-like dataset and train a multiclass logistic regression from scratch.

## Scripts

| Script | Purpose |
|--------|---------|
| `describe.py` | Summary statistics for numeric features |
| `histogram.py` | Feature histograms |
| `scatter_plot.py` | Scatter plots |
| `pair_plot.py` | Pairwise feature plots |
| `logreg_train.py` | Train logistic regression, save weights |
| `logreg_predict.py` | Predict houses from saved model |
| `dataset.py` / `utils.py` | Dataset helpers |

## Data

- `dataset_train.csv` — labeled training set
- `dataset_test.csv` — test set

## Usage

```bash
python describe.py dataset_train.csv
python histogram.py dataset_train.csv
python logreg_train.py dataset_train.csv
python logreg_predict.py dataset_test.csv weights.csv
```

Requires Python 3 with `numpy` and `pandas` (and a plotting stack for the viz scripts).
