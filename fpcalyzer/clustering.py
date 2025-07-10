import numpy as np
import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    KMeans,
    HDBSCAN,
    SpectralClustering,
    AgglomerativeClustering,
)
from skimage.color import lab2lch, deltaE_ciede2000
from sklearn.metrics import pairwise_distances, silhouette_score

from fpcalyzer.util.color_util import (
    clustering_score_lab_max,
    find_closest_names,
    get_named_colors,
)

from .util.color_util import delta_e_wrap


def kmeans(
    inks_df: pd.DataFrame,
    clusters=None,
    max_clusters=None,
    delta_e_max=None,
    random_state=None,
):
    if max_clusters is None:
        max_clusters = len(inks_df)

    if delta_e_max is None:
        delta_e_max = 25.0

    # print("kmeans", clusters, max_clusters, delta_e_max, random_state)

    cluster_range = range(1, int(max_clusters) + 1)
    if clusters is not None:
        cluster_range = [int(clusters)]

    lab = inks_df[["L", "a", "b"]].values
    colors = get_named_colors()

    lowest = float("inf")
    lowest_labels = None

    for n_c in cluster_range:
        clustering = KMeans(
            n_clusters=n_c,
            random_state=random_state,
        )
        clustering.fit(lab)
        score = clustering_score_lab_max(clustering, lab)
        if score < lowest:
            lowest = score
            lowest_labels = clustering.labels_
        # print(f"n_clusters={n_c}, mean ΔE00={score:.2f}")
        if score < float(delta_e_max):
            break

    inks_df["cluster"] = lowest_labels

    categories = inks_df[["L", "a", "b", "cluster"]].groupby(["cluster"]).mean()
    categories[["L_", "C_", "H_"]] = lab2lch(categories[["L", "a", "b"]].values)
    categories = categories.sort_values("H_")
    categories["name"] = find_closest_names(
        categories[["L", "a", "b"]].values,
        colors[["L_", "a_", "b_"]].values,
        colors["color_name"],
    )

    return inks_df, categories


def hdbscan(inks_df: pd.DataFrame, min_size=None, max_size=None):
    if min_size is None:
        min_size = 7

    if max_size is None:
        max_size = 100

    max_size = int(max_size)
    min_size = int(min_size)

    # print("hdbscan", min_size, max_size)

    lab = inks_df[["L", "a", "b"]].values
    colors = get_named_colors()
    n = len(lab)

    dists = np.zeros((n, n))
    for i in range(n):
        dists[i] = deltaE_ciede2000(lab[i][None, :], lab)
    clustering = HDBSCAN(
        min_cluster_size=min_size, max_cluster_size=max_size, metric="precomputed"
    )
    clustering.fit(dists)

    inks_df["cluster"] = clustering.labels_

    categories = inks_df[["L", "a", "b", "cluster"]].groupby(["cluster"]).mean()
    # print("UNCLASSIFIED", len(inks_df[inks_df["cluster"] == -1]))
    categories = categories[categories.index != -1]
    categories[["L_", "C_", "H_"]] = lab2lch(categories[["L", "a", "b"]].values)
    categories = categories.sort_values("H_")
    categories["name"] = find_closest_names(
        categories[["L", "a", "b"]].values,
        colors[["L_", "a_", "b_"]].values,
        colors["color_name"],
    )

    return inks_df, categories


def dbscan(inks_df: pd.DataFrame, delta_e_sim=None, min_size=None):
    if delta_e_sim is None:
        delta_e_sim = 4.0

    if min_size is None:
        min_size = 3

    delta_e_sim = float(delta_e_sim)
    min_size = int(min_size)

    # print("dbscan", delta_e_sim, min_size)

    lab = inks_df[["L", "a", "b"]].values
    colors = get_named_colors()
    n = len(lab)

    dists = np.zeros((n, n))
    for i in range(n):
        dists[i] = deltaE_ciede2000(lab[i][None, :], lab)
    clustering = DBSCAN(eps=delta_e_sim, min_samples=min_size, metric="precomputed")
    clustering.fit(dists)

    inks_df["cluster"] = clustering.labels_

    categories = inks_df[["L", "a", "b", "cluster"]].groupby(["cluster"]).mean()
    # print("UNCLASSIFIED", len(inks_df[inks_df["cluster"] == -1]))
    categories = categories[categories.index != -1]
    categories[["L_", "C_", "H_"]] = lab2lch(categories[["L", "a", "b"]].values)
    categories = categories.sort_values("H_")
    categories["name"] = find_closest_names(
        categories[["L", "a", "b"]].values,
        colors[["L_", "a_", "b_"]].values,
        colors["color_name"],
    )

    return inks_df, categories


def spectral(
    inks_df: pd.DataFrame,
    clusters=None,
    max_clusters=None,
    score_max=None,
    sigma=None,
    random_state=None,
):
    if max_clusters is None:
        max_clusters = len(inks_df) // 4

    if score_max is None:
        score_max = 0.25

    if sigma is None:
        sigma = 20.0

    if clusters is not None:
        clusters = int(clusters)

    max_clusters = int(max_clusters)
    score_max = float(score_max)
    sigma = float(sigma)

    # print("spectral", clusters, max_clusters, score_max, random_state)

    cluster_range = range(2, int(max_clusters) + 1)
    if clusters is not None:
        cluster_range = [int(clusters)]

    lab = inks_df[["L", "a", "b"]].values
    colors = get_named_colors()

    distance_matrix = pairwise_distances(lab.reshape(-1, 3), metric=delta_e_wrap)

    # Convert to affinity matrix (similarity): Gaussian kernel
    affinity_matrix = np.exp(-(distance_matrix**2) / (2 * sigma**2))

    lowest = float("inf")
    lowest_labels = None

    for n_c in cluster_range:
        clustering = SpectralClustering(
            n_clusters=n_c,
            random_state=random_state,
            affinity="precomputed",
            assign_labels="kmeans",
        )
        clustering.fit(affinity_matrix)
        score = silhouette_score(affinity_matrix, clustering.labels_)
        if score < lowest:
            lowest = score
            lowest_labels = clustering.labels_
        # print(f"n_clusters={n_c}, score={score:.2f}")
        if score < float(score_max):
            break

    inks_df["cluster"] = lowest_labels

    categories = inks_df[["L", "a", "b", "cluster"]].groupby(["cluster"]).mean()
    categories[["L_", "C_", "H_"]] = lab2lch(categories[["L", "a", "b"]].values)
    categories = categories.sort_values("H_")
    categories["name"] = find_closest_names(
        categories[["L", "a", "b"]].values,
        colors[["L_", "a_", "b_"]].values,
        colors["color_name"],
    )

    return inks_df, categories


def glom(
    inks_df: pd.DataFrame,
    clusters=None,
    max_clusters=None,
    score_max=None,
    random_state=None,
):
    if max_clusters is None:
        max_clusters = len(inks_df) // 4

    if score_max is None:
        score_max = 0.15

    # print("agglomerative", clusters, max_clusters, score_max, random_state)

    cluster_range = range(2, int(max_clusters) + 1)
    if clusters is not None:
        cluster_range = [int(clusters)]

    lab = inks_df[["L", "a", "b"]].values
    colors = get_named_colors()

    distance_matrix = pairwise_distances(lab.reshape(-1, 3), metric=delta_e_wrap)

    lowest = float("inf")
    lowest_labels = None

    for n_c in cluster_range:
        clustering = AgglomerativeClustering(
            n_clusters=n_c, metric="precomputed", linkage="complete"
        )
        clustering.fit(distance_matrix)
        score = silhouette_score(distance_matrix, clustering.labels_)
        if score < lowest:
            lowest = score
            lowest_labels = clustering.labels_
        # print(f"n_clusters={n_c}, score={score:.2f}")
        if score < float(score_max):
            break

    inks_df["cluster"] = lowest_labels

    categories = inks_df[["L", "a", "b", "cluster"]].groupby(["cluster"]).mean()
    categories[["L_", "C_", "H_"]] = lab2lch(categories[["L", "a", "b"]].values)
    categories = categories.sort_values("H_")
    categories["name"] = find_closest_names(
        categories[["L", "a", "b"]].values,
        colors[["L_", "a_", "b_"]].values,
        colors["color_name"],
    )

    return inks_df, categories
