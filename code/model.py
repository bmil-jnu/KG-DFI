"""
KG-DFI model definitions.

Architecture: KGEncoder (R-GCN) -> CrossAttention -> predictor head
(binary or multi-class). Matches the configuration used to produce the
results reported in Table 2 (binary) and Table 3 (multi-class).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from torch.utils.data import Dataset


class FDI_Dataset(Dataset):
    """(food_id, drug_id, label) triples for a single split."""

    def __init__(self, food_ids, drug_ids, labels):
        self.food_ids = torch.as_tensor(food_ids, dtype=torch.long)
        self.drug_ids = torch.as_tensor(drug_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.food_ids)

    def __getitem__(self, idx):
        return {"food_id": self.food_ids[idx], "drug_id": self.drug_ids[idx],
                "label": self.labels[idx]}


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.

    Binary task: `alpha` is a single scalar (config['focal_alpha'], default
        0.25 -- the standard value from Lin et al. 2017), applied as
        alpha_t = alpha if y=1 else (1-alpha).
    Multi-class task: `alpha` must be a per-class weight TENSOR
        (config['class_weights']), computed with sklearn's
        `compute_class_weight('balanced', ...)` raised to a power
        (`FOCAL_ALPHA_POWER = 1.5`, see dataloader.load_all_data) so that
        rare classes (Harmful, Negative) are weighted substantially more
        heavily than the abundant Possible class. This is NOT the same
        alpha=0.25 used for the binary task -- see README.md /
        "Focal loss class weighting" for why this matters when interpreting
        validation loss vs. validation F1 during multi-class training.
    """

    def __init__(self, task="binary", alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        assert task in ("binary", "multi")
        self.task = task
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.task == "binary":
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
            p_t = torch.where(targets == 1, inputs, 1 - inputs)
            if self.alpha is not None:
                alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            else:
                alpha_t = 1.0
            focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class KGEncoder(nn.Module):
    """R-GCN encoder over the heterogeneous drug-food knowledge graph.

    `num_bases=None` (the default, and what reproduces Table 3) gives each
    relation its own full weight matrix, matching the paper's description
    ("R-GCN assigns relation-specific weight matrices to different edge
    types"). Basis decomposition (`num_bases` = a small int) trades some of
    that per-relation specificity for fewer parameters; it is only used by
    the binary-task configuration in this repo's demo notebook.
    """

    def __init__(self, num_entities, num_relations, embedding_dim,
                 num_layers=3, num_bases=None, dropout=0.1):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        rgconv_kwargs = {"num_bases": num_bases} if num_bases is not None else {}
        self.rgcn_layers = nn.ModuleList([
            RelGraphConv(embedding_dim, embedding_dim, num_relations, **rgconv_kwargs,
                         activation=F.relu if i < num_layers - 1 else None)
            for i in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        nn.init.xavier_uniform_(self.entity_embeddings.weight)

    def forward(self, graph):
        h = self.entity_embeddings.weight
        for i, (rgcn, ln, drop) in enumerate(zip(self.rgcn_layers, self.layer_norms, self.dropouts)):
            h_new = drop(ln(rgcn(graph, h, graph.edata["type"])))
            h = h + h_new if i > 0 else h_new
        return h


class CrossAttention(nn.Module):
    """Food-to-drug cross-attention: food embeddings attend to drug embeddings."""

    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, food_emb, drug_emb):
        B = food_emb.size(0)
        Q = self.query(food_emb).view(B, self.num_heads, self.head_dim)
        K = self.key(drug_emb).view(B, self.num_heads, self.head_dim)
        V = self.value(drug_emb).view(B, self.num_heads, self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        attended = torch.matmul(attn_weights, V).reshape(B, -1)
        return self.layer_norm(food_emb + attended)


class MultiClassPredictor(nn.Module):
    """MLP head for the 4-class task (Possible / Positive / Negative / Harmful)."""

    def __init__(self, embedding_dim, num_classes=4, num_layers=3,
                 dropout=0.2, use_batch_norm=True):
        super().__init__()
        d = embedding_dim * 2
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.BatchNorm1d(d * 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d * 2, d), nn.BatchNorm1d(d) if use_batch_norm else nn.Identity(),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d, d // 2), nn.BatchNorm1d(d // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(), nn.Dropout(dropout / 2),
        )
        self.residual = nn.Linear(d, d // 2)
        self.out = nn.Sequential(
            nn.Linear(d // 2, d // 4), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(d // 4, num_classes),
        )

    def forward(self, food_emb, drug_emb):
        x = torch.cat([food_emb, drug_emb], dim=1)
        return self.out(self.net(x) + self.residual(x))


class BinaryPredictor(nn.Module):
    """MLP head for the binary interaction-presence task (Table 2)."""

    def __init__(self, embedding_dim, hidden_dim=64, num_layers=4, dropout=0.2,
                 use_batch_norm=True):
        super().__init__()
        interaction_dim = embedding_dim * 2
        dims = [interaction_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(dims[i + 1]) if use_batch_norm else nn.Identity()
            for i in range(len(dims) - 2)
        )
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(len(dims) - 2))

    def forward(self, food_emb, drug_emb):
        x = torch.cat([food_emb, drug_emb], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            x_new = F.relu(self.batch_norms[i](layer(x)))
            x_new = self.dropouts[i](x_new)
            x = x + x_new if i > 0 and x.shape == x_new.shape else x_new
        return torch.sigmoid(self.layers[-1](x)).squeeze(-1)


class KGDFI(nn.Module):
    """Full KG-DFI model: KGEncoder -> CrossAttention -> predictor head."""

    def __init__(self, num_entities, num_relations, config):
        super().__init__()
        task = config["task"]
        assert task in ("binary", "multi")
        self.task = task
        embedding_dim = config.get("embedding_dim", 256)
        num_heads = config.get("num_heads", 4)

        self.encoder = KGEncoder(
            num_entities, num_relations, embedding_dim,
            num_layers=config.get("kg_layers", 3),
            num_bases=config.get("kg_bases"),
            dropout=config.get("kg_dropout", 0.1),
        )
        self.use_attention = config.get("use_attention", True)
        if self.use_attention:
            self.attention = CrossAttention(embedding_dim, num_heads)

        if task == "multi":
            self.predictor = MultiClassPredictor(
                embedding_dim, num_classes=config.get("num_classes", 4),
                num_layers=config.get("pred_layers", 3),
                dropout=config.get("pred_dropout", 0.2),
                use_batch_norm=config.get("use_batch_norm", True),
            )
        else:
            self.predictor = BinaryPredictor(
                embedding_dim, num_layers=config.get("pred_layers", 4),
                dropout=config.get("pred_dropout", 0.2),
                use_batch_norm=config.get("use_batch_norm", True),
            )

    def forward(self, graph, food_ids, drug_ids):
        emb = self.encoder(graph)
        food_emb, drug_emb = emb[food_ids], emb[drug_ids]
        if self.use_attention:
            food_emb = self.attention(food_emb, drug_emb)
        return self.predictor(food_emb, drug_emb)


def create_fdi_model(config: dict, num_entities: int, num_relations: int, device="cpu") -> KGDFI:
    """Factory used by the demo notebook: build a KGDFI model from a flat config dict.

    Required key: 'task' ('binary' or 'multi'). All other keys have the
    defaults listed in `KGDFI.__init__` / the individual submodules.
    """
    model = KGDFI(num_entities, num_relations, config)
    return model.to(device)


def get_model_info(model: KGDFI) -> dict:
    """Summary used by `utils.print_model_info`: parameter counts and task."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "task": model.task,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "size_mb": total * 4 / (1024 ** 2),  # float32
        "uses_attention": model.use_attention,
    }
