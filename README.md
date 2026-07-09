[README.md](https://github.com/user-attachments/files/29848162/README.md)
# KG-DFI: Knowledge Graph-based Drug-Food Interaction Prediction

KG-DFI predicts drug-food interactions (DFIs) by representing each food as a
single entity in a heterogeneous biomedical knowledge graph (FooDB + DRKG +
CTD), rather than decomposing it into its constituent compounds. An R-GCN
encoder learns entity representations from the graph, and a food-to-drug
cross-attention module models the interaction between a specific food and
drug pair before a final MLP predicts interaction presence (binary) or type
(Possible / Positive / Negative / Harmful).

## Repository layout

```
KG-DFI/
├── code/
│   ├── model.py        # KGEncoder, CrossAttention, predictors, FocalLoss, create_fdi_model()
│   ├── dataloader.py    # load_all_data() -- graph + train/val/test DataLoaders
│   ├── train.py         # train_model(), save_checkpoint(), load_checkpoint()
│   ├── evaluate.py       # evaluate_model(), save_results(), bootstrap_ci()
│   └── utils.py          # seed setting, device selection, printing, plotting
├── data/
│   └── README.md         # data sources, expected file layout, download links
├── KG-DFI.ipynb           # end-to-end demo: Part 1 (binary, Table 2), Part 2 (multi-class, Table 3)
├── environment.yaml
└── README.md              # this file
```

## Setup

```bash
conda env create -f environment.yaml
conda activate kgdfi
```

Core dependencies: Python 3.9, PyTorch 2.2, DGL 2.4, scikit-learn, pandas,
numpy, tqdm.

## Data

The processed knowledge graph, entity/relation mappings, train/val/test
splits, and pretrained model checkpoints are hosted on Zenodo
(DOI: [10.5281/zenodo.21275098](https://doi.org/10.5281/zenodo.21275098))
and are not stored in this repository due to size. See
[`data/README.md`](data/README.md) for the expected directory layout. In
brief, `data_dir` (passed to `load_all_data`) must contain:

- `background_kg_graph.pt` — DGL graph (41,698 entities, 71 relation types;
  direct drug-food edges removed to prevent leakage)
- `phase1_results.json` — `{"num_entities": ..., "num_relations": ...}`
- `mappings.json` — `{"node_to_id": {entity_name: node_index}}`
- `{train,val,test}_raw.csv` — DFI pairs for the binary task
- a multi-class labeled CSV (see `data/README.md`) for the 4-class task

See [`data/KG_CONSTRUCTION.md`](data/KG_CONSTRUCTION.md) for how
`background_kg_graph.pt` was built from FooDB, DRKG, and CTD (data sources,
integration steps, and the connectivity-based filtering criterion applied to
CTD), including guidance for rebuilding the graph from a different database
snapshot.

## Training

Run [`KG-DFI.ipynb`](KG-DFI.ipynb) end to end. It has two self-contained parts:

- **Part 1 (cells 2-7)** — binary interaction-presence task, reproduces Table 2.
- **Part 2 (cells 8-13)** — 4-class task (Possible/Positive/Negative/Harmful),
  reproduces Table 3, using `multi_config` (embedding_dim=256, 3 R-GCN layers,
  4 attention heads, focal-loss gamma=1.0, no basis decomposition) — the
  configuration selected via the 144-configuration grid search described in
  Supplementary File 5.

Each part's model-initialization cell is immediately followed by an
"optional: skip training" cell. Leave `LOAD_PRETRAINED_BINARY` /
`LOAD_PRETRAINED_MULTI` at `False` to train from scratch, or set it to `True`
to instead load the released checkpoint (`best_model_binary.pt` /
`best_model.pt`, see [`data/README.md`](data/README.md)) via
`train.load_checkpoint` and skip straight to evaluation.

Each part follows the same four-step pattern, which you can also call
directly from your own script instead of the notebook:

```python
from dataloader import load_all_data
from model import create_fdi_model
from train import train_model, save_checkpoint
from evaluate import evaluate_model, save_results

data = load_all_data(task="multi", data_dir="data/", batch_size=64)
data["graph"] = data["graph"].to(device)
if data["class_weights"] is not None:
    multi_config["class_weights"] = data["class_weights"].to(device)

model = create_fdi_model(multi_config, num_entities=data["mappings"]["num_nodes"],
                          num_relations=data["mappings"]["num_relations"], device=device)
model, history = train_model(model, data["train_loader"], data["val_loader"],
                              data["graph"], multi_config, device=device)
save_checkpoint(model, "results/multiclass")

results = evaluate_model(model, data["test_loader"], data["graph"], device,
                          task="multi", num_classes=4)
save_results(results, "results/multiclass", "test_results.json")
```

To skip training and evaluate the released checkpoint instead, replace the
`train_model` call with:

```python
from train import load_checkpoint
model = load_checkpoint(model, "data/best_model.pt", device=device)
```

`evaluate_model` also accepts a `bootstrap_ci`-style re-analysis: call
`evaluate.bootstrap_ci(y_true, y_pred, y_proba, task="multi")` on the saved
`results["predictions"]` arrays to get a 95% confidence interval for every
metric via 10,000 bootstrap resamples, without retraining.

### Focal loss class weighting

The binary model uses a fixed `alpha=0.25` (the standard default from Lin et
al., 2017). The multi-class model instead uses a **per-class** weight vector,
computed as `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`
raised to the power 1.5, so that the rare Harmful (2.4% of samples) and
Negative (5.1%) classes are weighted substantially more heavily than the
abundant Possible class (73.7%) — see `train.py::_make_criterion`. Model
selection and early stopping are based on validation-set weighted F1 rather
than validation loss, since the class-imbalance-aware weighting can cause
validation loss to rise even as F1 improves (a small number of heavily
weighted rare-class errors can dominate the aggregate loss while the
argmax-based classification boundary continues to improve).

## Evaluation

`code/evaluate.py` exposes:
- `run_inference(model, graph, loader, device)` — collect labels/predictions/probabilities
- `evaluate_model(model, test_loader, graph, device, task, num_classes)` — full
  metrics dict (including the confusion matrix) + predictions, ready for `save_results`
- `compute_binary_metrics` / `compute_multiclass_metrics` — accuracy, F1,
  AUROC, AUPRC, Cohen's kappa (and per-class recall for the multi-class task)
- `bootstrap_ci(y_true, y_pred, y_proba, task)` — 95% CI via 10,000 bootstrap resamples

`bootstrap_ci` is not called automatically by `train_model`/`evaluate_model` --
call it directly on a saved checkpoint's predictions when you want confidence
intervals without retraining (as in the analysis behind Table 2/3's reported CIs).

## Reproducibility notes

- All reported results use `random_state=42` for the stratified train/val/test
  split (70/15/15) and for model weight initialization.
- The knowledge graph is constructed once (see `data/README.md` for the
  FooDB/DRKG/CTD integration pipeline) with all direct drug-food interaction
  edges removed to prevent label leakage.
- The model operates in a transductive setting: all entities, including those
  in the test split, contribute to embedding learning during training. See
  the manuscript's Limitations section for the corresponding caveat regarding
  cold-start generalization.

## Citation

If you use this code, please cite:

> Kang, M., Lee, M. J., & Yoo, S. KG-DFI: A prediction of drug-food
> interactions based on knowledge graph embedding. *Computational and
> Structural Biotechnology Journal* (in press).

## Contact

If you have any questions or comments, please feel free to create an issue on github here, or email us:

- kmg013858@gmail.com
- syyoo@jnu.ac.kr
