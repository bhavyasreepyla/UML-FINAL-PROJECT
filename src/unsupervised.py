"""Unsupervised compute library: corpus stats, TF-IDF, embeddings, clustering.

Library-only. All compute logic lives here; matplotlib figures live in
`src/visualization.py`. Driver in `scripts/run_eda.py` wires the two.

Inputs:
    Preprocessed DataFrame from `data.load_dataframe` (or any frame with the
    same `Title`/`text`/`clean_combined`/`combined_short`/`User_Needs` columns),
    and optionally precomputed (N, D) embedding arrays.

Outputs:
    Fitted models and pandas results. The only file-touching helper is
    `load_or_compute_mini_embeddings`, which proxies to `embeddings.py` and
    caches its result there. No disk I/O elsewhere in this module.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    CLASS_COLORS,
    DATA_PATH,
    EMBEDDINGS_MINI_PATH,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_SELECTION_METHOD,
    KMEANS_K_RANGE,
    LABEL_COLUMN,
    LABELED_CLASSES,
    N_TOPICS_NMF,
    RANDOM_STATE,
    TEXT_COLUMN,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TITLE_COLUMN,
    UMAP_METRIC,
    UMAP_MIN_DIST_CLUSTER,
    UMAP_MIN_DIST_VIZ,
    UMAP_N_COMPONENTS_CLUSTER,
    UMAP_N_COMPONENTS_VIZ,
    UMAP_N_NEIGHBORS,
    UNLABELED_VALUE,
)


# Data loading + corpus statistics


def load_eda_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the preprocessed dataset and add EDA-only convenience columns.

    In:  path to the preprocessed EDA CSV.
    Out: DataFrame with extra columns: `text_len_chars`, `text_len_words`,
         `title_len_words`, `is_labeled`.
    """
    from data import load_dataframe

    df = load_dataframe(path)
    df["text_len_chars"] = df[TEXT_COLUMN].str.len()
    df["text_len_words"] = df[TEXT_COLUMN].str.split().str.len()
    df["title_len_words"] = df[TITLE_COLUMN].str.split().str.len()
    df["is_labeled"] = df[LABEL_COLUMN] != UNLABELED_VALUE
    return df


def corpus_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and print corpus-wide text statistics.

    In:  DataFrame from `load_eda_data`.
    Out: 1-column DataFrame of stats indexed by metric name; also printed.
    """
    stats = {
        "total_articles": len(df),
        "labeled": df["is_labeled"].sum(),
        "unlabeled": (~df["is_labeled"]).sum(),
        "pct_unlabeled": f"{(~df['is_labeled']).mean()*100:.1f}%",
        "text_len_mean": df["text_len_chars"].mean(),
        "text_len_median": df["text_len_chars"].median(),
        "text_len_std": df["text_len_chars"].std(),
        "text_len_max": df["text_len_chars"].max(),
        "text_len_min": df["text_len_chars"].min(),
        "word_count_mean": df["text_len_words"].mean(),
        "word_count_median": df["text_len_words"].median(),
        "word_count_max": df["text_len_words"].max(),
    }
    stats_df = pd.DataFrame([stats]).T
    stats_df.columns = ["value"]
    print("\n=== Corpus Statistics ===")
    print(stats_df.to_string())
    return stats_df


def print_imbalance_ratios(df: pd.DataFrame) -> None:
    """Print each class's count + ratio relative to the largest class.

    In:  DataFrame with `is_labeled` + label column.
    Out: stdout printout; no return value.
    """
    df_lab = df[df["is_labeled"]]
    counts = df_lab[LABEL_COLUMN].value_counts()
    max_c = counts.max()
    print("\n=== Class Imbalance Ratios (relative to largest class) ===")
    for cls, cnt in counts.items():
        print(f"  {cls:25s}  {cnt:5d}  ratio = 1:{max_c/cnt:.1f}")


# TF-IDF and topic modeling


def fit_tfidf(
    texts: Iterable[str],
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    max_df: float = TFIDF_MAX_DF,
) -> Tuple[TfidfVectorizer, "pd.api.extensions.ExtensionArray"]:
    """Fit a TF-IDF vectorizer on a corpus.

    In:  iterable of texts; standard TF-IDF knobs.
    Out: (vectorizer, sparse matrix).
    """
    vec = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range,
        sublinear_tf=True, stop_words="english",
        min_df=min_df, max_df=max_df,
    )
    matrix = vec.fit_transform(texts)
    return vec, matrix


def fit_nmf(tfidf_matrix, tfidf_vectorizer: TfidfVectorizer,
            n_topics: int = N_TOPICS_NMF) -> Tuple[NMF, np.ndarray, Dict[int, List[str]]]:
    """Fit NMF topic model and extract the top words per topic.

    In:  TF-IDF sparse matrix; fitted vectorizer; number of topics.
    Out: (nmf_model, topic_matrix (N, n_topics), {topic_id: top words list}).
    """
    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE, max_iter=400)
    topic_matrix = nmf.fit_transform(tfidf_matrix)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    top_words = {}
    for i, topic in enumerate(nmf.components_):
        top_words[i] = [feature_names[j] for j in topic.argsort()[-12:][::-1]]
    return nmf, topic_matrix, top_words


def compute_nmf_topic_alignment(df: pd.DataFrame, text_col: str = "clean_combined"):
    """Fit NMF over the corpus and crosstab topic vs `User_Needs` (labelled only).

    In:  DataFrame with the text column + label column + `is_labeled`.
    Out: (top_words dict, crosstab DataFrame row-normalised by class).
    """
    tfidf_vec, tfidf_matrix = fit_tfidf(df[text_col])
    _, topic_matrix, top_words = fit_nmf(tfidf_matrix, tfidf_vec)
    df = df.copy()
    df["nmf_topic"] = topic_matrix.argmax(axis=1)
    df_lab = df[df["is_labeled"]]
    ct = pd.crosstab(df_lab[LABEL_COLUMN], df_lab["nmf_topic"], normalize="index")
    return top_words, ct


# N-grams and vocabulary distance


def top_ngrams_per_class(
    df: pd.DataFrame, n_top: int = 15, ngram_range: Tuple[int, int] = (1, 1)
) -> Dict[str, List[Tuple[str, float]]]:
    """Per-class top TF-IDF n-grams from `clean_combined`.

    In:  labelled-included df; how many top terms; ngram range.
    Out: dict {class: [(term, mean_tfidf), ...]} for each labelled class.
    """
    df_lab = df[df["is_labeled"]]
    results: Dict[str, List[Tuple[str, float]]] = {}
    for cls in LABELED_CLASSES:
        texts = df_lab[df_lab[LABEL_COLUMN] == cls]["clean_combined"].values
        if len(texts) == 0:
            continue
        vec = TfidfVectorizer(
            max_features=5000, ngram_range=ngram_range,
            sublinear_tf=True, stop_words="english", min_df=2,
        )
        X = vec.fit_transform(texts)
        mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
        top_idx = mean_tfidf.argsort()[-n_top:][::-1]
        features = vec.get_feature_names_out()
        results[cls] = [(features[i], round(mean_tfidf[i], 4)) for i in top_idx]
    return results


def compute_class_vocab_distributions(
    df: pd.DataFrame, max_features: int = 5000
) -> Dict[str, np.ndarray]:
    """Per-class normalized term-frequency distribution for JSD computation.

    In:  df with `clean_combined` + label column.
    Out: dict {class: probability vector over the shared vocabulary}.
    """
    df_lab = df[df["is_labeled"]]
    vec = CountVectorizer(
        max_features=max_features, stop_words="english", min_df=3, ngram_range=(1, 2)
    )
    vec.fit(df_lab["clean_combined"])

    distributions = {}
    for cls in LABELED_CLASSES:
        texts = df_lab[df_lab[LABEL_COLUMN] == cls]["clean_combined"]
        X = vec.transform(texts)
        freq = np.asarray(X.sum(axis=0)).flatten().astype(float)
        freq += 1e-10  # Laplace smoothing
        freq /= freq.sum()
        distributions[cls] = freq
    return distributions


def compute_jsd_matrix(distributions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Jensen–Shannon divergence between class vocabulary distributions.

    In:  dict {class: probability vector}.
    Out: DataFrame of pairwise JSD values, symmetric.
    """
    classes = list(distributions.keys())
    n = len(classes)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            jsd = jensenshannon(distributions[classes[i]], distributions[classes[j]])
            matrix[i, j] = jsd
            matrix[j, i] = jsd
    return pd.DataFrame(matrix, index=classes, columns=classes)


def detect_near_duplicates(
    df: pd.DataFrame, threshold: float = 0.95, sample_size: int = 5000
) -> pd.DataFrame:
    """TF-IDF cosine duplicate detection; returns pairs above `threshold`.

    In:  df with `clean_combined`; cosine cutoff; max sample for memory.
    Out: DataFrame of pairs with similarity + labels + title previews.
    """
    df_sample = (
        df.sample(sample_size, random_state=RANDOM_STATE) if len(df) > sample_size else df
    )
    _, X = fit_tfidf(df_sample["clean_combined"], max_features=10_000, min_df=2)

    pairs = []
    batch_size = 500
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        sim_block = cosine_similarity(X[start:end], X)
        for i in range(end - start):
            global_i = start + i
            for j in range(global_i + 1, X.shape[0]):
                if sim_block[i, j] >= threshold:
                    pairs.append({
                        "idx_a": df_sample.index[global_i],
                        "idx_b": df_sample.index[j],
                        "similarity": round(sim_block[i, j], 4),
                        "label_a": df_sample.iloc[global_i][LABEL_COLUMN],
                        "label_b": df_sample.iloc[j][LABEL_COLUMN],
                        "title_a": str(df_sample.iloc[global_i][TITLE_COLUMN])[:80],
                        "title_b": str(df_sample.iloc[j][TITLE_COLUMN])[:80],
                    })

    dup_df = pd.DataFrame(pairs)
    print(f"\n=== Near-Duplicate Detection (threshold={threshold}) ===")
    print(f"  Checked: {len(df_sample):,} articles")
    print(f"  Near-duplicate pairs found: {len(dup_df):,}")
    if len(dup_df) > 0:
        cross = dup_df["label_a"] != dup_df["label_b"]
        print(f"  Cross-label duplicates: {cross.sum()}")
        lab_unlab = (
            ((dup_df["label_a"] == UNLABELED_VALUE) & (dup_df["label_b"] != UNLABELED_VALUE))
            | ((dup_df["label_b"] == UNLABELED_VALUE) & (dup_df["label_a"] != UNLABELED_VALUE))
        )
        print(f"  Labeled↔Unlabeled duplicates: {lab_unlab.sum()}")
    return dup_df


# Embeddings (thin wrapper around embeddings.py)


def load_or_compute_mini_embeddings(df: pd.DataFrame, force: bool = False) -> np.ndarray:
    """Return cached MiniLM embeddings or compute them via `embeddings.py`.

    In:  df with `combined_short`; `force` recomputes even when cache exists.
    Out: (N, 384) MiniLM embeddings.
    """
    import os

    if os.path.isfile(EMBEDDINGS_MINI_PATH) and not force:
        print(f"Loading cached embeddings from {EMBEDDINGS_MINI_PATH}")
        return np.load(EMBEDDINGS_MINI_PATH)
    from embeddings import compute_sbert_embeddings

    return compute_sbert_embeddings(
        df["combined_short"].tolist(), model_key="mini", force_recompute=force
    )


# Dimensionality reduction


def fit_umap(embeddings: np.ndarray, n_components: int, min_dist: float):
    """Fit UMAP and project embeddings.

    In:  (N, D) embeddings; target dimension; min_dist hyperparameter.
    Out: (reducer, (N, n_components) projected embeddings).
    """
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=n_components,
        min_dist=min_dist,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    return reducer, reducer.fit_transform(embeddings)


def compute_umap_projections(embeddings: np.ndarray):
    """Compute both 5D-for-clustering and 2D-for-viz UMAP projections.

    In:  (N, D) embeddings.
    Out: (reducer_5d, reduced_5d, reducer_2d, reduced_2d).
    """
    r5, d5 = fit_umap(embeddings, UMAP_N_COMPONENTS_CLUSTER, UMAP_MIN_DIST_CLUSTER)
    r2, d2 = fit_umap(embeddings, UMAP_N_COMPONENTS_VIZ, UMAP_MIN_DIST_VIZ)
    return r5, d5, r2, d2


def compute_umap_sweep(
    embeddings: np.ndarray,
    n_neighbors_list: Iterable[int] = (5, 10, 15),
    metric_list: Iterable[str] = ("correlation", "euclidean", "cosine"),
) -> Dict[Tuple[int, str], np.ndarray]:
    """Compute a grid of UMAP-2D projections across (n_neighbors × metric).

    In:  (N, D) embeddings; iterables of n_neighbors and metric values.
    Out: dict {(n_neighbors, metric): (N, 2) projection}.
    """
    out: Dict[Tuple[int, str], np.ndarray] = {}
    for n_neighbors in n_neighbors_list:
        for metric in metric_list:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors, n_components=2,
                metric=metric, random_state=RANDOM_STATE,
            )
            out[(n_neighbors, metric)] = reducer.fit_transform(embeddings)
    return out


def compute_tsne_projection(
    embeddings: np.ndarray, sample_size: int = 3000
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a sample of embeddings into 2D via t-SNE.

    In:  (N, D) embeddings; max sample size (t-SNE scales poorly).
    Out: (indices used (M,), (M, 2) projection).
    """
    from sklearn.manifold import TSNE

    n = len(embeddings)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca")
    return idx, tsne.fit_transform(embeddings[idx])


# Clustering


def fit_hdbscan(reduced_embeddings: np.ndarray):
    """Run HDBSCAN on reduced embeddings and print a cluster summary.

    In:  (N, k) UMAP projection (typically 5D).
    Out: (HDBSCAN clusterer, labels (N,) — -1 means noise).
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_SELECTION_METHOD,
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"HDBSCAN: {n_clusters} clusters, "
          f"{n_noise} noise ({n_noise / len(labels) * 100:.1f}%)")
    return clusterer, labels


def fit_kmeans(reduced_embeddings: np.ndarray, k: int):
    """Fit K-Means with a chosen K.

    In:  (N, k_dim) reduced embeddings; integer K.
    Out: (KMeans model, cluster labels (N,)).
    """
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    return km, km.fit_predict(reduced_embeddings)


def kmeans_sweep(reduced_embeddings: np.ndarray, k_range=KMEANS_K_RANGE):
    """K-Means inertia + silhouette across `k_range`; picks best K by silhouette.

    In:  (N, k_dim) reduced embeddings; iterable of K candidates.
    Out: (dict with `k` / `inertia` / `silhouette` lists, best_k integer).
    """
    results = {"k": [], "inertia": [], "silhouette": []}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = km.fit_predict(reduced_embeddings)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(silhouette_score(reduced_embeddings, labels))
    best_k = results["k"][np.argmax(results["silhouette"])]
    print(f"Best K by silhouette: {best_k} (score={max(results['silhouette']):.4f})")
    return results, best_k


def build_bertopic(sbert_model=None):
    """Assemble a BERTopic model with project sub-components (UMAP/HDBSCAN/cTF-IDF).

    In:  optional pre-loaded SentenceTransformer; defaults to MiniLM.
    Out: configured (not yet fit) BERTopic model.
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer

    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS_CLUSTER,
        min_dist=UMAP_MIN_DIST_CLUSTER, metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_SELECTION_METHOD,
    )
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = KeyBERTInspired()

    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    return BERTopic(
        embedding_model=sbert_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        verbose=True,
    )


def fit_bertopic_on_corpus(
    df: pd.DataFrame, embeddings: np.ndarray, text_col: str = "combined_short"
):
    """Fit BERTopic on the full corpus with precomputed embeddings.

    In:  df with `is_labeled` + text column + label column; (N, D) embeddings.
    Out: (topic_model, topics (list, length N), keywords dict for top topics,
          crosstab of label × top-12 topics row-normalised).
    """
    print("  Running BERTopic on full corpus (labeled + unlabeled)...")
    topic_model = build_bertopic()
    topics, _ = topic_model.fit_transform(df[text_col].tolist(), embeddings=embeddings)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = (np.array(topics) == -1).sum()
    print(
        f"  BERTopic: {n_topics} topics, {n_outliers} outliers "
        f"({n_outliers/len(topics)*100:.1f}%)"
    )

    df_with = df.copy()
    df_with["bertopic_topic"] = topics
    df_lab = df_with[df_with["is_labeled"]]
    top_topics = (
        df_lab[df_lab["bertopic_topic"] != -1]["bertopic_topic"]
        .value_counts()
        .head(12)
        .index.tolist()
    )
    df_lab_filtered = df_lab[df_lab["bertopic_topic"].isin(top_topics)]
    ct = pd.crosstab(
        df_lab_filtered[LABEL_COLUMN], df_lab_filtered["bertopic_topic"], normalize="index"
    )

    topic_info = topic_model.get_topic_info()
    keywords: Dict[int, Tuple[int, List[str]]] = {}
    for _, row in topic_info.head(13).iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(tid)[:6]]
        keywords[tid] = (int(row["Count"]), words)

    print("\n  BERTopic-Label alignment (top-1 topic concentration per class):")
    for cls in LABELED_CLASSES:
        if cls in ct.index:
            top_topic = ct.loc[cls].idxmax()
            print(f"    {cls:25s} -> Topic {top_topic} ({ct.loc[cls].max():.1%})")

    return topic_model, topics, keywords, ct


def evaluate_clustering(
    embeddings: np.ndarray, labels: np.ndarray, ground_truth_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute silhouette (excluding noise) + optional ARI/NMI vs ground truth.

    In:  embeddings (or reduced coords); cluster labels; optional GT labels.
    Out: dict with `silhouette` and (if GT provided) `ari`, `nmi`.
    """
    metrics: Dict[str, float] = {}
    non_noise = labels != -1
    if non_noise.sum() > 1 and len(set(labels[non_noise])) > 1:
        metrics["silhouette"] = silhouette_score(embeddings[non_noise], labels[non_noise])
    if ground_truth_labels is not None:
        metrics["ari"] = adjusted_rand_score(ground_truth_labels, labels)
        metrics["nmi"] = normalized_mutual_info_score(ground_truth_labels, labels)
    return metrics


def evaluate_all_clusterings(
    df: pd.DataFrame, reduced_5d: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Run HDBSCAN + K-Means (k=6) and score both vs ground-truth `User_Needs`.

    In:  df with `is_labeled` + label column; (N, 5) UMAP-for-clustering.
    Out: (metrics_df indexed by algorithm name with ARI/NMI/Silhouette/n_clusters/n_noise,
          dict of {algorithm: cluster labels (N,)}).
    """
    _, hdb_labels = fit_hdbscan(reduced_5d)
    _, km6_labels = fit_kmeans(reduced_5d, k=6)

    labeled_mask = df["is_labeled"].values
    gt_labels = df.loc[labeled_mask, LABEL_COLUMN].values

    metrics: Dict[str, Dict[str, float]] = {}
    for name, labels in [("HDBSCAN", hdb_labels), ("K-Means (k=6)", km6_labels)]:
        lab_subset = labels[labeled_mask]
        m: Dict[str, float] = {
            "ARI": adjusted_rand_score(gt_labels, lab_subset),
            "NMI": normalized_mutual_info_score(gt_labels, lab_subset),
        }
        non_noise = lab_subset != -1
        if non_noise.sum() > 1 and len(set(lab_subset[non_noise])) > 1:
            m["Silhouette"] = silhouette_score(
                reduced_5d[labeled_mask][non_noise], lab_subset[non_noise]
            )
        m["n_clusters"] = len(set(labels)) - (1 if -1 in labels else 0)
        m["n_noise"] = int((labels == -1).sum())
        metrics[name] = m

    metrics_df = pd.DataFrame(metrics).T
    print("\n=== Clustering vs. Ground Truth ===")
    print(metrics_df.to_string(float_format="{:.4f}".format))
    return metrics_df, {"HDBSCAN": hdb_labels, "K-Means (k=6)": km6_labels}


def cluster_sample_titles(df: pd.DataFrame, cluster_col: str,
                          title_col: str = TITLE_COLUMN, n_samples: int = 3) -> None:
    """Print a few representative article titles per cluster.

    In:  df with cluster + title columns; column names; samples per cluster.
    Out: stdout printout; no return value.
    """
    for cl in sorted(df[cluster_col].unique()):
        if cl == -1:
            continue
        subset = df[df[cluster_col] == cl]
        print(f"\nCluster {cl} ({len(subset)} articles):")
        samples = subset[title_col].sample(
            min(n_samples, len(subset)), random_state=RANDOM_STATE)
        for t in samples.values:
            print(f"  - {t[:100]}")


# Centroids and pseudo-labelling


def compute_class_centroids(df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """Mean embedding per labelled class.

    In:  df aligned with `embeddings`; (N, D) embeddings.
    Out: dict {class: mean embedding (D,)}.
    """
    centroids: Dict[str, np.ndarray] = {}
    for cls in LABELED_CLASSES:
        mask = (df[LABEL_COLUMN] == cls).values
        if mask.sum() > 0:
            centroids[cls] = embeddings[mask].mean(axis=0)
    return centroids


def compute_centroid_similarity_matrix(centroids: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise cosine-similarity heatmap between class centroids.

    In:  dict {class: centroid vector} from `compute_class_centroids`.
    Out: square DataFrame of cosine similarities (1.0 on the diagonal).
    """
    classes = list(centroids.keys())
    matrix = np.vstack([centroids[c] for c in classes])
    sim = cosine_similarity(matrix)
    return pd.DataFrame(sim, index=classes, columns=classes)


def identify_pseudo_label_candidates(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    top_n: int = 50,
    confidence_threshold: float = 0.85,
) -> pd.DataFrame:
    """Rank unlabelled articles by max cosine similarity to labelled centroids.

    In:  df aligned with `embeddings`; per-class top-N to keep; high-confidence cutoff.
    Out: DataFrame of candidates with `pseudo_label`, `pseudo_confidence`, sorted by
         confidence. Prints distribution + counts.
    """
    centroids = compute_class_centroids(df, embeddings)
    unlabeled_mask = ~df["is_labeled"].values
    unlabeled_emb = embeddings[unlabeled_mask]
    unlabeled_df = df[unlabeled_mask].copy().reset_index(drop=True)

    centroid_matrix = np.vstack([centroids[c] for c in LABELED_CLASSES])
    sim = cosine_similarity(unlabeled_emb, centroid_matrix)
    unlabeled_df["pseudo_label"] = [LABELED_CLASSES[i] for i in sim.argmax(axis=1)]
    unlabeled_df["pseudo_confidence"] = sim.max(axis=1)

    candidates = [
        unlabeled_df[unlabeled_df["pseudo_label"] == cls].nlargest(top_n, "pseudo_confidence")
        for cls in LABELED_CLASSES
    ]
    candidates_df = pd.concat(candidates).sort_values("pseudo_confidence", ascending=False)

    high_conf = candidates_df[candidates_df["pseudo_confidence"] >= confidence_threshold]
    print(f"\n=== Pseudo-Label Candidates ===")
    print(f"  Total candidates: {len(candidates_df)}")
    print(f"  High-confidence (≥{confidence_threshold}): {len(high_conf)}")
    print(f"  Distribution: {high_conf['pseudo_label'].value_counts().to_dict()}")
    return candidates_df
