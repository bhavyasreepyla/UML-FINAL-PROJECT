"""All matplotlib plots in one place: EDA figures + supervised training plots.

Library-only. Every plot function accepts an optional `save_path`; when None
the figure is shown interactively and not persisted. Compute logic lives in
`src/unsupervised.py` and `src/supervised.py` — these plotters take
already-computed arrays / DataFrames as input.

Inputs:
    Arrays, DataFrames, and dicts produced by compute helpers. `plot_training_log`
    reads `logs/<run_name>.jsonl`.

Outputs:
    PNG files at `save_path` when provided; otherwise `plt.show()`.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from config import (
    CLASS_COLORS,
    LABEL_COLUMN,
    LABELED_CLASSES,
    LOG_DIR,
    TEXT_COLUMN,
    TITLE_COLUMN,
    UNLABELED_VALUE,
)


def _save_or_show(fig, save_path: Optional[str], dpi: int = 200) -> None:
    """Persist a figure to `save_path` (creating parent dirs) or show it."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# Corpus / EDA distribution plots


def plot_text_length_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """Full-corpus length histogram + per-class box plot + per-class word-count violin.

    In:  df with `text_len_chars`, `text_len_words`, `is_labeled`, label column.
    Out: figure saved / shown.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.hist(df["text_len_chars"], bins=80, color="#4a90d9", edgecolor="white", alpha=0.85)
    median, mean = df["text_len_chars"].median(), df["text_len_chars"].mean()
    ax.axvline(median, color="#e74c3c", ls="--", lw=2, label=f"Median: {median:,.0f}")
    ax.axvline(mean, color="#f39c12", ls="--", lw=2, label=f"Mean: {mean:,.0f}")
    ax.set_xlabel("Article Length (characters)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Text Length Distribution (All Articles)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    df_labeled = df[df["is_labeled"]].copy()
    order = (
        df_labeled.groupby(LABEL_COLUMN)["text_len_chars"]
        .median().sort_values().index.tolist()
    )
    colors = [CLASS_COLORS.get(c, "#999") for c in order]

    ax = axes[1]
    bp = ax.boxplot(
        [df_labeled[df_labeled[LABEL_COLUMN] == c]["text_len_chars"].values for c in order],
        labels=order, vert=True, patch_artist=True, showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Characters", fontsize=12)
    ax.set_title("Text Length by Class (no outliers)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[2]
    sns.violinplot(
        data=df_labeled, x=LABEL_COLUMN, y="text_len_words",
        order=order, palette=CLASS_COLORS, inner="quartile", cut=0, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Words", fontsize=12)
    ax.set_title("Word Count by Class", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Text length distribution saved")


def plot_class_imbalance(df: pd.DataFrame, save_path: Optional[str] = None):
    """Label-count bar chart + labelled-vs-unlabelled pie.

    In:  df with `is_labeled` + label column.
    Out: figure saved / shown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    counts = df[LABEL_COLUMN].value_counts()
    colors = [CLASS_COLORS.get(c, "#999") for c in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(val + 20, bar.get_y() + bar.get_height() / 2, f"{val:,}",
                va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Articles", fontsize=12)
    ax.set_title("Label Distribution (incl. 'none')", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    ax = axes[1]
    labeled_n = df["is_labeled"].sum()
    unlabeled_n = (~df["is_labeled"]).sum()
    ax.pie(
        [labeled_n, unlabeled_n],
        labels=[f"Labeled\n({labeled_n:,})", f"Unlabeled\n({unlabeled_n:,})"],
        colors=["#4a90d9", "#cccccc"],
        autopct="%1.1f%%", textprops={"fontsize": 12},
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Labeled vs. Unlabeled Split", fontsize=13, fontweight="bold")

    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Class imbalance saved")


def plot_monthly_publications(df: pd.DataFrame, save_path: Optional[str] = None):
    """Stacked bar of articles per month, segmented by `User_Needs`.

    In:  df with `Publish_date` + label column.
    Out: figure saved / shown.
    """
    out = df.copy()
    out["Publish_month"] = (
        pd.to_datetime(out["Publish_date"], errors="coerce").dt.strftime("%Y-%m")
    )
    out = out.dropna(subset=["Publish_month"])

    counts = (
        out.groupby(["Publish_month", LABEL_COLUMN]).size()
        .unstack(fill_value=0).sort_index()
    )
    column_order = [c for c in [UNLABELED_VALUE] + LABELED_CLASSES if c in counts.columns]
    counts = counts[column_order]
    colors = [CLASS_COLORS.get(c, "#999") for c in counts.columns]

    fig, ax = plt.subplots(figsize=(14, 6))
    counts.plot(kind="bar", stacked=True, color=colors, ax=ax, edgecolor="white", width=0.85)
    ax.set_xlabel("Month")
    ax.set_ylabel("Articles published")
    ax.set_title("Monthly publication volume by User_Needs", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Monthly publications saved")


def plot_word_clouds(df: pd.DataFrame, save_path: Optional[str] = None):
    """Per-class word clouds drawn from `clean_combined`.

    In:  df with `clean_combined` + label column.
    Out: figure saved / shown. No-op if `wordcloud` package missing.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("✗ Word clouds skipped (wordcloud package not installed)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, cls in enumerate(LABELED_CLASSES):
        ax = axes[i]
        subset = df[df[LABEL_COLUMN] == cls]
        text = " ".join(subset["clean_combined"].dropna().astype(str).tolist())
        if not text.strip():
            ax.set_visible(False)
            continue
        wc = WordCloud(
            width=500, height=300, background_color="white",
            max_words=80, colormap="Dark2",
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(cls, fontsize=13, fontweight="bold",
                     color=CLASS_COLORS.get(cls, "#444"))
        ax.axis("off")

    for j in range(len(LABELED_CLASSES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Word clouds by User_Needs", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Word clouds saved")


# N-gram / vocabulary plots


def _plot_ngrams_grid(ngrams: Dict[str, List[Tuple[str, float]]], title: str, save_path: Optional[str]):
    """Internal: 6-panel barh of per-class top n-grams."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for i, cls in enumerate(LABELED_CLASSES):
        ax = axes[i]
        if cls not in ngrams:
            ax.set_visible(False)
            continue
        words, scores = zip(*ngrams[cls])
        ax.barh(range(len(words)), scores, color=CLASS_COLORS[cls], edgecolor="white", alpha=0.85)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(cls, fontsize=13, fontweight="bold", color=CLASS_COLORS[cls])
        ax.set_xlabel("Mean TF-IDF", fontsize=10)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_ngram_analysis(
    unigrams: Dict[str, List[Tuple[str, float]]],
    bigrams: Dict[str, List[Tuple[str, float]]],
    save_path_unigram: Optional[str] = None,
    save_path_bigram: Optional[str] = None,
):
    """Two 6-panel figures: top unigrams and top bigrams per class.

    In:  pre-computed dicts from `unsupervised.top_ngrams_per_class`.
    Out: two figures (unigrams + bigrams), saved / shown.
    """
    _plot_ngrams_grid(unigrams, "Top Unigrams per Class (TF-IDF)", save_path_unigram)
    _plot_ngrams_grid(bigrams, "Top Bigrams per Class (TF-IDF)", save_path_bigram)
    print("✓ N-gram analysis saved (unigrams + bigrams)")


def plot_vocabulary_overlap(jsd_df: pd.DataFrame, save_path: Optional[str] = None):
    """JSD heatmap + hierarchical clustering dendrogram.

    In:  pre-computed JSD matrix DataFrame.
    Out: figure saved / shown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    mask = np.triu(np.ones_like(jsd_df, dtype=bool), k=1)
    sns.heatmap(
        jsd_df, annot=True, fmt=".3f", cmap="YlOrRd_r", mask=mask, ax=ax,
        vmin=0, vmax=0.5, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Jensen–Shannon Divergence"},
    )
    ax.set_title("Vocabulary Overlap (JSD)\nLower = More Similar", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)

    ax = axes[1]
    condensed = []
    classes = list(jsd_df.index)
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            condensed.append(jsd_df.iloc[i, j])
    Z = linkage(condensed, method="average")
    dendrogram(Z, labels=classes, ax=ax,
               leaf_rotation=35, leaf_font_size=11, color_threshold=0.15)
    ax.set_title("Class Vocabulary Clustering", fontsize=13, fontweight="bold")
    ax.set_ylabel("JSD Distance")

    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Vocabulary overlap (JSD) saved")


# Embedding / clustering plots


def plot_embedding_space(df: pd.DataFrame, reduced_2d: np.ndarray, save_path: Optional[str] = None):
    """UMAP-2D scatter: all articles by label, then labelled-only.

    In:  df with label column; (N, 2) UMAP projection.
    Out: figure saved / shown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes[0]
    for cls in [UNLABELED_VALUE] + LABELED_CLASSES:
        mask = (df[LABEL_COLUMN] == cls).values
        ax.scatter(
            reduced_2d[mask, 0], reduced_2d[mask, 1],
            c=CLASS_COLORS.get(cls, "#ccc"), label=cls,
            s=3 if cls == UNLABELED_VALUE else 8,
            alpha=0.2 if cls == UNLABELED_VALUE else 0.6,
        )
    ax.set_title("UMAP — All Articles by User_Needs", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, fontsize=9, loc="best")

    ax = axes[1]
    for cls in LABELED_CLASSES:
        mask = (df[LABEL_COLUMN] == cls).values
        ax.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                   c=CLASS_COLORS[cls], label=cls, s=12, alpha=0.6)
    ax.set_title("UMAP — Labeled Articles Only", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=3, fontsize=9, loc="best")

    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Embedding space (UMAP) saved")


def plot_umap_sweep(
    projections: Dict[Tuple[int, str], np.ndarray],
    df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Grid of UMAP-2D projections from a hyperparameter sweep.

    In:  dict {(n_neighbors, metric): (N, 2) projection} from
         `unsupervised.compute_umap_sweep`; df aligned with embeddings.
    Out: figure saved / shown.
    """
    n_neighbors_set = sorted({k[0] for k in projections})
    metric_set = sorted({k[1] for k in projections})
    fig, axes = plt.subplots(
        len(n_neighbors_set), len(metric_set),
        figsize=(5 * len(metric_set), 5 * len(n_neighbors_set)),
    )

    labels = df[LABEL_COLUMN].astype(str).values
    for i, n in enumerate(n_neighbors_set):
        for j, metric in enumerate(metric_set):
            ax = axes[i, j] if len(n_neighbors_set) > 1 else axes[j]
            proj = projections[(n, metric)]
            for cls in [UNLABELED_VALUE] + LABELED_CLASSES:
                mask = labels == cls
                if not mask.any():
                    continue
                ax.scatter(
                    proj[mask, 0], proj[mask, 1],
                    c=CLASS_COLORS.get(cls, "#ccc"),
                    s=3 if cls == UNLABELED_VALUE else 6,
                    alpha=0.2 if cls == UNLABELED_VALUE else 0.6,
                    label=cls if i == 0 and j == 0 else None,
                )
            ax.set_title(f"n_neighbors={n}, metric={metric}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    handles, leg_labels = (
        axes.flatten()[0].get_legend_handles_labels()
        if hasattr(axes, "flatten") else axes[0].get_legend_handles_labels()
    )
    fig.legend(handles, leg_labels, loc="upper center",
               ncol=len(handles), fontsize=9, markerscale=3)
    fig.suptitle("UMAP hyperparameter sweep", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ UMAP sweep saved")


def plot_tsne_vs_umap(
    idx: np.ndarray,
    tsne_proj: np.ndarray,
    reduced_2d: np.ndarray,
    df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Side-by-side t-SNE and UMAP-2D coloured by label.

    In:  `idx` sample indices used for t-SNE; `tsne_proj` (M, 2); the full
         UMAP-2D (N, 2); df with label column.
    Out: figure saved / shown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    labels = df[LABEL_COLUMN].astype(str).values
    for ax, proj, title in [
        (axes[0], tsne_proj, f"t-SNE (n={len(idx)})"),
        (axes[1], reduced_2d[idx], "UMAP (same sample)"),
    ]:
        sub_labels = labels[idx]
        for cls in [UNLABELED_VALUE] + LABELED_CLASSES:
            mask = sub_labels == cls
            if not mask.any():
                continue
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=CLASS_COLORS.get(cls, "#ccc"),
                s=4 if cls == UNLABELED_VALUE else 10,
                alpha=0.3 if cls == UNLABELED_VALUE else 0.7,
                label=cls,
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    axes[1].legend(markerscale=3, fontsize=9, loc="best")
    fig.suptitle("t-SNE vs UMAP — labelled corpus projection",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ t-SNE vs UMAP comparison saved")


def plot_centroid_distances(sim_df: pd.DataFrame, save_path: Optional[str] = None):
    """Pairwise cosine-similarity heatmap between class centroids.

    In:  square similarity DataFrame from
         `unsupervised.compute_centroid_similarity_matrix`.
    Out: figure saved / shown.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(sim_df, dtype=bool), k=1)
    sns.heatmap(
        sim_df, annot=True, fmt=".3f", cmap="RdYlGn", mask=mask, ax=ax,
        vmin=0.5, vmax=1.0, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_title(
        "Class Centroid Similarity (Embedding Space)\nHigher = Harder to Separate",
        fontsize=13, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Centroid distances saved")


def plot_clustering_evaluation(
    df: pd.DataFrame,
    reduced_2d: np.ndarray,
    cluster_labels_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """N+1 panel scatter: each clustering algorithm + ground truth, on UMAP-2D.

    In:  df with label column; (N, 2) UMAP; dict {algorithm: labels (N,)}.
    Out: figure saved / shown.
    """
    n = len(cluster_labels_dict)
    fig, axes = plt.subplots(1, n + 1, figsize=(8 * (n + 1), 7))

    for ax, (name, labels) in zip(axes[:n], cluster_labels_dict.items()):
        scatter = ax.scatter(reduced_2d[:, 0], reduced_2d[:, 1],
                             c=labels, cmap="Spectral", s=3, alpha=0.4)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        plt.colorbar(scatter, ax=ax, label="Cluster")

    ax = axes[n]
    for cls in [UNLABELED_VALUE] + LABELED_CLASSES:
        mask = (df[LABEL_COLUMN] == cls).values
        ax.scatter(
            reduced_2d[mask, 0], reduced_2d[mask, 1],
            c=CLASS_COLORS.get(cls, "#ccc"), label=cls,
            s=3 if cls == UNLABELED_VALUE else 8,
            alpha=0.2 if cls == UNLABELED_VALUE else 0.5,
        )
    ax.set_title("Ground Truth Labels", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, fontsize=8, loc="best")

    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Clustering evaluation saved")


def plot_clusters_vs_ground_truth(
    reduced_2d: np.ndarray,
    cluster_labels: np.ndarray,
    ground_truth: np.ndarray,
    class_names: Iterable[str],
    cluster_title: str = "Clusters",
    gt_title: str = "Ground Truth",
    save_path: Optional[str] = None,
):
    """Side-by-side scatter of one clustering vs ground-truth labels.

    In:  (N, 2) coords; cluster labels (N,); GT labels (N,); class-name list.
    Out: figure saved / shown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    scatter = axes[0].scatter(
        reduced_2d[:, 0], reduced_2d[:, 1],
        c=cluster_labels, cmap="Spectral", s=3, alpha=0.5)
    axes[0].set_title(cluster_title, fontsize=14)
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
    plt.colorbar(scatter, ax=axes[0], label="Cluster")

    le = LabelEncoder()
    encoded_gt = le.fit_transform(ground_truth)
    axes[1].scatter(reduced_2d[:, 0], reduced_2d[:, 1],
                    c=encoded_gt, cmap="tab10", s=3, alpha=0.5)
    axes[1].set_title(gt_title, fontsize=14); axes[1].set_xlabel("UMAP-1")
    class_names = list(class_names)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=plt.cm.tab10(i / len(class_names)),
                   markersize=8, label=c)
        for i, c in enumerate(class_names)
    ]
    axes[1].legend(handles=handles, loc="best", fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_path, dpi=150)


def plot_kmeans_sweep(
    results_dict: Dict[str, List],
    reference_k: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Elbow + silhouette curves across a K sweep, with optional `reference_k` marker.

    In:  dict with `k` / `inertia` / `silhouette` lists; optional vertical-line K.
    Out: figure saved / shown.
    """
    ks = results_dict["k"]
    inertias = results_dict["inertia"]
    silhouettes = results_dict["silhouette"]
    best_k = ks[int(np.argmax(silhouettes))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ks, inertias, "o-", color="#4a90d9", lw=2, markersize=6)
    if reference_k is not None:
        axes[0].axvline(reference_k, color="#e74c3c", ls="--", alpha=0.7,
                        label=f"k={reference_k} (reference)")
        axes[0].legend()
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Plot", fontsize=13, fontweight="bold")

    axes[1].plot(ks, silhouettes, "o-", color="#2ca02c", lw=2, markersize=6)
    if reference_k is not None:
        axes[1].axvline(reference_k, color="#e74c3c", ls="--", alpha=0.7,
                        label=f"k={reference_k} (reference)")
    axes[1].axvline(best_k, color="#f39c12", ls="--", alpha=0.7,
                    label=f"best k={best_k}")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs. K", fontsize=13, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    _save_or_show(fig, save_path)
    print(f"✓ K-Means sweep saved (best k={best_k})")


# Topic-modeling alignment plots


def _plot_topic_alignment(
    ct: pd.DataFrame,
    keyword_lines: List[str],
    main_title: str,
    keyword_title: str,
    save_path: Optional[str],
):
    """Internal: heatmap of label × topic crosstab + monospace keyword listing."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues",
                ax=axes[0], linewidths=0.5, linecolor="white")
    axes[0].set_title(main_title, fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Topic")
    axes[0].set_ylabel("User_Needs Class")

    axes[1].axis("off")
    axes[1].text(
        0.02, 0.95, "\n\n".join(keyword_lines), transform=axes[1].transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
    )
    axes[1].set_title(keyword_title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_topic_label_alignment(
    ct: pd.DataFrame,
    top_words: Dict[int, List[str]],
    save_path: Optional[str] = None,
):
    """NMF topic distribution per class + topic-keyword listing.

    In:  pre-computed crosstab (rows=label, cols=topic_id, row-normalised);
         dict {topic_id: top words}.
    Out: figure saved / shown.
    """
    lines = [f"Topic {i}: {', '.join(words[:8])}" for i, words in top_words.items()]
    _plot_topic_alignment(
        ct, lines,
        main_title="NMF Topic Distribution per Class\n(row-normalized)",
        keyword_title="NMF Topic Keywords",
        save_path=save_path,
    )
    print("✓ NMF topic-label alignment saved")


def plot_bertopic_analysis(
    ct: pd.DataFrame,
    keywords: Dict[int, Tuple[int, List[str]]],
    save_path: Optional[str] = None,
):
    """BERTopic topic × class crosstab + topic-keyword listing.

    In:  pre-computed crosstab (rows=label, cols=topic_id, row-normalised);
         keywords dict {topic_id: (doc_count, [top words])}.
    Out: figure saved / shown.
    """
    lines = [
        f"Topic {tid} ({count} docs): {', '.join(words)}"
        for tid, (count, words) in list(keywords.items())[:12]
    ]
    _plot_topic_alignment(
        ct, lines,
        main_title="BERTopic topics vs User_Needs classes\n(row-normalized, top-12 topics)",
        keyword_title="BERTopic topic keywords (top-12)",
        save_path=save_path,
    )
    print("✓ BERTopic topic-label alignment saved")


# Pseudo-label plot


def plot_pseudo_label_candidates(
    df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    reduced_2d: np.ndarray,
    save_path: Optional[str] = None,
):
    """UMAP highlight of high-confidence pseudo-label candidates.

    In:  full df with label column; candidates DataFrame from
         `unsupervised.identify_pseudo_label_candidates`; (N, 2) UMAP.
    Out: figure saved / shown.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c="#eeeeee", s=2, alpha=0.3)

    for cls in LABELED_CLASSES:
        mask = (df[LABEL_COLUMN] == cls).values
        ax.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                   c=CLASS_COLORS[cls], s=8, alpha=0.4, label=cls)

    high_conf = candidates_df[candidates_df["pseudo_confidence"] >= 0.85]
    if len(high_conf) > 0:
        print(f"  Highlighting {len(high_conf)} high-confidence candidates")

    ax.set_title("Pseudo-Label Candidates on Embedding Space",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=3, fontsize=8, loc="best")
    plt.tight_layout()
    _save_or_show(fig, save_path)
    print("✓ Pseudo-label candidates saved")


# Supervised pipeline plots (kept from prior version)


def plot_model_comparison(results_df: pd.DataFrame, title: str = "Model Comparison",
                          save_path: Optional[str] = None):
    """Horizontal-bar chart of acc / F1-macro / F1-weighted per model.

    In:  DataFrame indexed by model name with columns acc/f1_macro/f1_weighted.
    Out: figure saved / shown.
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(results_df) * 0.8)))
    results_df[["f1_macro", "f1_weighted", "acc"]].plot(
        kind="barh", ax=ax, color=["#2196F3", "#4CAF50", "#FF9800"])
    ax.set_title(title, fontsize=14); ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save_or_show(fig, save_path, dpi=150)


def plot_confusion_matrix(y_true, y_pred, label_names,
                          title: str = "Confusion Matrix", save_path: Optional[str] = None):
    """Standard confusion-matrix heatmap.

    In:  true labels; predicted labels; class names.
    Out: figure saved / shown.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    _save_or_show(fig, save_path, dpi=150)


def plot_prediction_distribution(df_top3: pd.DataFrame, save_path: Optional[str] = None):
    """Top-1 tag-frequency bar + top-1 confidence histogram for unlabeled predictions.

    In:  top-K DataFrame with `tag_1` and `confidence_1` columns.
    Out: figure saved / shown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tag_counts = df_top3["tag_1"].value_counts()
    tag_counts.plot(kind="bar", ax=axes[0], color="#2196F3", edgecolor="black")
    axes[0].set_title("Top-1 Predicted Tag Distribution")
    axes[0].set_xlabel("Predicted Tag"); axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].hist(df_top3["confidence_1"], bins=30,
                 color="#4CAF50", edgecolor="black", alpha=0.8)
    axes[1].set_title("Top-1 Confidence Score Distribution")
    axes[1].set_xlabel("Confidence"); axes[1].set_ylabel("Count")
    median = df_top3["confidence_1"].median()
    axes[1].axvline(x=median, color="red", linestyle="--", label=f"Median: {median:.3f}")
    axes[1].legend()
    plt.tight_layout()
    _save_or_show(fig, save_path, dpi=150)


def plot_training_log(run_name: str, log_dir: str = LOG_DIR, save_path: Optional[str] = None):
    """Read `logs/<run_name>.jsonl` and line-plot each numeric metric over runs.

    In:  run_name (filename stem written by trainers); log dir.
    Out: figure saved / shown; prints a message and returns if no log.
    """
    log_path = os.path.join(log_dir, f"{run_name}.jsonl")
    if not os.path.isfile(log_path):
        print(f"No log found at {log_path}"); return

    entries = []
    with open(log_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    if not entries:
        print("Empty log file"); return

    df = pd.DataFrame(entries)
    skip = {"run_name", "base_model", "model_id", "saved_at",
            "elapsed_seconds", "train_size", "batch_size",
            "num_epochs", "num_iterations"}
    metric_cols = [c for c in df.columns if c not in skip]
    if not metric_cols:
        print("No metric columns found"); return

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in metric_cols:
        if df[col].dtype in (float, int, np.float64, np.int64):
            ax.plot(range(len(df)), df[col], "o-", label=col)
    ax.set_xlabel("Run"); ax.set_ylabel("Score")
    ax.set_title(f"Training Progress: {run_name}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path, dpi=150)


def plot_tensorboard_hint(run_name: str, log_dir: str = LOG_DIR) -> None:
    """Print the `tensorboard --logdir ...` command for a given run."""
    tb_dir = os.path.join(log_dir, "tensorboard", run_name)
    print(f"For detailed training curves, run:\n  tensorboard --logdir {tb_dir}")
