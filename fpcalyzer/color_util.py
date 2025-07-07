from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage.color import deltaE_ciede2000, rgb2lab


DATA_PATH = Path(__file__).parent / "data"


@lru_cache
def get_named_colors(
    fname: str = "colors.csv", compute_lab: bool = True
) -> pd.DataFrame:
    """Load color file to DataFrame

    Args:
        fname (str, optional): Filename of the CSV in ./data. Defaults to "colors.csv".
        compute_lab (bool, optional): If the Lab color columns need to be computed from RGB. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with color_name, hex, r, g, b, L_, a_, b_ columns
    """
    colors_file = DATA_PATH / fname
    names_df = pd.read_csv(colors_file)
    if compute_lab:
        rgbs = names_df[["r", "g", "b"]].to_numpy() / 255.0
        names_df[["L_", "a_", "b_"]] = rgb2lab(rgbs).reshape(-1, 3)
    return names_df


@lru_cache(2)
def get_fpc_data(user_id: int) -> dict:
    """Get an FPC user's ink data JSON

    Args:
        user_id (int): FPC user ID

    Returns:
        dict: FPC API's user data return
    """
    r = httpx.get(
        f"https://www.fountainpencompanion.com/users/{user_id}",
        headers={
            "Accept": "application/vnd.api+json",
            "User-Agent": "fpcalyzer-backend",
        },
    )
    data = r.json()
    return data


def fpc_to_df(fpc_data: dict) -> pd.DataFrame:
    """Interpret the FPC user's data into a DataFrame of inks and colors

    Args:
        fpc_data (dict): FPC API's user data (probably from `get_fpc_data`)

    Returns:
        pd.DataFrame: A DataFrame with brand_name, color, comment, ink_id, ink_name, kind, line_name, maker, R, G, B, label, L, a, and b columns.
    """
    included = {x["id"]: x["attributes"] for x in fpc_data["included"]}
    collected_inks = [
        included[x["id"]]
        for x in fpc_data["data"]["relationships"]["collected_inks"]["data"]
    ]

    inks_df = pd.DataFrame(collected_inks)
    inks_df = inks_df[inks_df["color"].notna() & (inks_df["color"] != "")]
    inks_df["R"] = inks_df["color"].str[1:3].apply(lambda x: int(x, 16))
    inks_df["G"] = inks_df["color"].str[3:5].apply(lambda x: int(x, 16))
    inks_df["B"] = inks_df["color"].str[5:7].apply(lambda x: int(x, 16))

    inks_df["label"] = (
        inks_df["brand_name"] + " " + inks_df["line_name"] + " " + inks_df["ink_name"]
    )
    inks_df = inks_df.drop_duplicates(subset="label", keep="first")

    rgb_normalized = inks_df[["R", "G", "B"]].to_numpy() / 255.0
    lab = rgb2lab(rgb_normalized).reshape(-1, 3)
    inks_df[["L", "a", "b"]] = lab

    return inks_df


def find_closest_names(lab1: np.ndarray, lab2: np.ndarray, names2: pd.Series) -> list:
    """Pair lab points to their closest color names.

    Args:
        lab1 (np.ndarray): Array of colors to assign names to
        lab2 (np.ndarray): Array of colors to match from
        names2 (pd.Series): Array of color names

    Returns:
        list: Per-assignee list of assigned names
    """
    closest_names = []
    for color in lab1:
        distances = np.array(
            [
                deltaE_ciede2000(color[np.newaxis, :], ref[np.newaxis, :])[0]
                for ref in lab2
            ]
        )
        closest_index = np.argmin(distances)
        closest_names.append(names2.iloc[closest_index])
    return closest_names


def delta_e_wrap(u: np.ndarray, v: np.ndarray) -> float:
    """Wrapper for the skimage.color.deltaE_ciede2000 function so clustering algorithms can use it.

    Args:
        u (np.ndarray): Lab point 1
        v (np.ndarray): Lab point 2

    Returns:
        float: The deltaE between the points
    """
    return deltaE_ciede2000(u[np.newaxis, :], v[np.newaxis, :])[0]


def clustering_score_lab(model: KMeans, X_lab: np.ndarray) -> float:
    """Calculate the average per-cluster deltaE of a KMeans model

    Args:
        model (KMeans): Post-fit KMeans object
        X_lab (np.ndarray): Lab points to fit       

    Returns:
        float: Average per-cluster delta E
    """
    centers_lab = model.cluster_centers_
    labels = model.labels_
    total_delta_e = 0.0
    for i, x in enumerate(X_lab):
        c = centers_lab[labels[i]]
        total_delta_e += deltaE_ciede2000(x[np.newaxis, :], c[np.newaxis, :])[0]
    return total_delta_e / len(X_lab)  # lower is better


def clustering_score_lab_max(model: KMeans, X_lab: np.ndarray) -> float:
    """Calculate the maximum per-cluster deltaE of a KMeans model

    Args:
        model (KMeans): Post-fit KMeans object
        X_lab (np.ndarray): Lab points to fit       

    Returns:
        float: Maximum per-cluster delta E
    """
    centers_lab = model.cluster_centers_
    labels = model.labels_
    max_delta_e = 0.0
    for i, x in enumerate(X_lab):
        c = centers_lab[labels[i]]
        max_delta_e = max(
            deltaE_ciede2000(x[np.newaxis, :], c[np.newaxis, :])[0], max_delta_e
        )
    return max_delta_e
