# Data

## Sources

The knowledge graph integrates three public databases:

- **FooDB** — food composition data (food-compound associations)
- **DRKG** (Drug Repurposing Knowledge Graph) — drug-gene/disease/side-effect/ATC relations
- **CTD** (Comparative Toxicogenomics Database) — compound-gene mechanistic interactions,
  filtered to edges where at least one endpoint already appears in the FooDB+DRKG graph

Drug-food interaction labels for training/evaluation come from **DDID**
(Diet-Drug Interactions Database).

## Processed files (required for `load_all_data`)

Download the processed data bundle from Zenodo
([https://doi.org/10.5281/zenodo.21275098](https://doi.org/10.5281/zenodo.21275098))
and extract it into this `data/` directory so that `data_dir='data/'` (as used
in `KG-DFI.ipynb`) resolves to a folder containing:

| File | Description |
|---|---|
| `background_kg_graph.pt` | DGL graph, 41,698 entities / 71 relation types, direct drug-food edges removed |
| `phase1_results.json` | `{"num_entities": 41698, "num_relations": 71}` |
| `mappings.json` | `{"node_to_id": {entity_name: node_index}}` |
| `train_raw.csv`, `val_raw.csv`, `test_raw.csv` | binary DFI pairs (70/15/15 stratified split, seed=42) |
| `fdi_multi_class_dataset.csv` | 4-class labeled DFI pairs (Possible/Positive/Negative/Harmful) |

## Pretrained model weights (optional)

The same Zenodo deposit also includes the trained checkpoints used to produce
the manuscript's reported results:

| File | Reproduces |
|---|---|
| `best_model.pt` | Table 3 (4-class model: embedding_dim=256, 3 R-GCN layers, gamma=1.0) |
| `best_model_binary.pt` | Table 2 (binary model) |

These are not required to run training, but let you skip straight to
evaluation. In `KG-DFI.ipynb`, set `LOAD_PRETRAINED_BINARY = True` (Part 1)
or `LOAD_PRETRAINED_MULTI = True` (Part 2) in the cell immediately before the
training cell; this calls `train.load_checkpoint(model, path, device=device)`
to load the released weights into the freshly constructed model instead of
training from scratch. The model passed to `load_checkpoint` must have been
constructed with `create_fdi_model` using the same `binary_config` /
`multi_config` used to produce the checkpoint (already the notebook default).

Note that the two checkpoints were originally saved in different formats
(`best_model.pt` as a plain `model.state_dict()`; `best_model_binary.pt` as a
dict with a `'model_state_dict'` key alongside training metadata) --
`load_checkpoint` detects and handles both automatically.

## Hosting

Model checkpoints and the processed-data bundle are hosted externally due to
GitHub's file size limits, via the Zenodo deposit above
(DOI: [10.5281/zenodo.21275098](https://doi.org/10.5281/zenodo.21275098)).

If you use this dataset, please cite:

> Kang, M., Lee, M. J., & Yoo, S. (2026). KG-DFI processed data and model
> checkpoints [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21275098
