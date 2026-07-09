# Knowledge Graph Construction

This document describes how `background_kg_graph.pt` (the processed graph
used by `load_all_data`, see [`README.md`](README.md)) was built from the
three source databases. It is provided as methodological documentation
rather than a runnable pipeline, since exact reconstruction depends on the
specific release/download snapshot of each source database (FooDB, DRKG,
and CTD are all updated periodically, and IDs are not guaranteed stable
across releases).

## 1. Source databases and their roles

| Source | Contributes | Entity types touched |
|---|---|---|
| **FooDB** | Food-compound composition data | `Food::*`, `Compound::*` |
| **DRKG** (Drug Repurposing Knowledge Graph) | Drug-gene, drug-disease, drug-side-effect, drug-ATC relations | `Compound::*` (drugs), `Gene::*`, disease/side-effect/ATC entities |
| **CTD** (Comparative Toxicogenomics Database) | Compound-gene mechanistic interactions | `Compound::*`, `Gene::*` |

Drug-food interaction (DFI) labels used for model training/evaluation come
from a separate source, **DDID**, and are *not* part of the background
knowledge graph (see Section 4).

## 2. Step 1 -- FooDB preprocessing

FooDB categorizes botanical ingredients into "foods" (conventional dietary
staples) and "herbs" (plant-derived materials typically administered in
concentrated/medicinal forms). Only the **"foods"** category is used; herbs
are excluded (see manuscript Methods, Section 2.1, for the rationale).

After filtering to the "foods" category, each food's constituent compounds
are recorded as `Food::<FoodID> --[CONTAINS]--> Compound::<CompoundID>`
triples. This step yields **98,603 unique food-compound triples, spanning
894 food entities and 11,580 unique compounds**. `CONTAINS` is the *only*
relation type that touches `Food::` entities anywhere in the final graph --
foods have no other connectivity into the knowledge graph.

## 3. Step 2 -- FooDB + DRKG integration

FooDB and DRKG are merged through **shared compound entities**, matched via
standardized chemical identifiers (PubChem CID, ChEMBL ID, DrugBank ID).
Concretely: a compound appearing in both FooDB (as a food constituent) and
DRKG (e.g. as a drug, or as a target of a drug-gene relation) is represented
as the *same* `Compound::*` node in the merged graph, so that food-derived
compounds and drug-related entities are directly connected wherever a shared
identifier exists.

This merge produces a baseline graph of **1,362,957 triplets**.

## 4. Step 3 -- CTD integration (connectivity-based filtering)

CTD contributes fine-grained compound-gene mechanistic edges (e.g.
`CTD::increases_expression::Compound:Gene`,
`CTD::decreases_activity::Compound:Gene`, etc. -- 21 unique CTD relation
subtypes in the final graph). The raw CTD extract contains 422,276 curated
compound-gene-disease interactions, far more than can be usefully integrated
without either (a) introducing many disconnected entities that contribute
nothing to DFI prediction, or (b) diluting the graph with mechanistic detail
for compounds/genes unrelated to any food or drug of interest.

**Filtering criterion**: a CTD triple is retained only if **at least one of
its two entities (the compound or the gene) already exists** in the
FooDB+DRKG baseline graph from Step 2. This keeps the graph connected and
task-relevant, at the cost of excluding CTD interactions between two
entities that are both otherwise absent from FooDB/DRKG (see manuscript
Discussion for the resulting bias toward well-characterized entities, e.g.
CYP450 enzymes).

This yields **85,010 CTD triples retained** (out of 422,276), covering 429
unique compounds and 7,468 unique genes with mechanistic annotations.

Beyond CTD, food-derived compounds additionally reach `Gene::*` entities
through **GNBR** (literature-mined relations, e.g. `GNBR::B::Compound:Gene`)
and, at the gene-drug boundary, **DGIDB** (e.g.
`DGIDB::ALLOSTERIC MODULATOR::Gene:Compound`) -- both are part of the DRKG
integration in Step 3, not separately filtered.

## 5. Final graph

| | |
|---|---|
| Total triples | 1,447,967 (98,603 FooDB + 1,264,354 DRKG + 85,010 CTD) |
| Entities | 41,698, across 7 categories (Food, Compound, Gene, Disease, Side Effect, ATC code, ...) |
| Relation types | 71 unique (142 bidirectional) |

Entities and relations are serialized as raw `(head, relation, tail)` triples
(tab-separated, columns `head`, `relation`, `tail`) before being converted
into the DGL graph object (`background_kg_graph.pt`) and the
`node_to_id` index (`mappings.json`) used by `dataloader.load_all_data`.
Naming convention: `Food::<ID>`, `Compound::<ID>` (shared by food constituents
and drugs), `Gene::<EntrezID>`, etc.

## 6. Preventing data leakage

**All direct drug-food interaction edges (from DDID) are excluded from the
background knowledge graph.** The graph therefore only encodes indirect,
mechanistic connectivity (food -> compound -> gene -> drug, etc.); any
drug-food interaction the model predicts must be inferred through this
indirect structure rather than memorized from a direct edge. DFI labels
(from DDID) are loaded separately, at training time, by
`dataloader._load_binary_splits` / `_load_multiclass_splits` (see
[`README.md`](README.md)).

## 7. Reproducing this pipeline from scratch

Because FooDB, DRKG, and CTD are each updated independently and entity IDs
are not guaranteed stable across releases, we do not provide a single
"one-click" reconstruction script; instead we release the exact processed
graph (`background_kg_graph.pt`) used to produce the manuscript's reported
results, together with the construction methodology above. Researchers
wishing to rebuild the graph from a different database snapshot should:

1. Download FooDB and filter to the "foods" category (excluding "herbs").
2. Download DRKG and merge with FooDB via shared PubChem CID / ChEMBL / DrugBank identifiers.
3. Download CTD and apply the connectivity-based filter described in Section 4.
4. Remove any direct drug-food interaction edges before training (Section 6).
