import binascii
import logging
from collections import defaultdict
from io import BytesIO
from json import JSONDecodeError
from flask import Flask, render_template, request, make_response, send_file

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2lch, lab2rgb, deltaE_ciede2000
from sklearn.manifold import TSNE

from .util import start_stopwatch, stop_stopwatch
from .util.color_util import (
    get_named_colors,
    get_fpc_data,
    fpc_to_df,
)
from .clustering import kmeans, hdbscan, dbscan, spectral, glom


def create_app():
    app = Flask(__name__)
    matplotlib.use("agg")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    @app.route("/")
    def root():
        """Send the index html"""
        return send_file("templates/index.html")


    @app.route("/favicon.ico")
    def favicon():
        """Send the favicon"""
        return send_file("templates/img/favicon.ico", mimetype="image/x-icon")


    @app.route("/analyze/<int:user_id>")
    def analyze(user_id: int):
        """Perform a clustering analysis on an FPC user

        Args:
            user_id (int): The user ID to analyze
        """
        method = request.args.get("method", "kmeans")
        try:
            fpc_data = get_fpc_data(user_id)
        except JSONDecodeError:
            return make_response("Not Found", 404)

        inks_df = fpc_to_df(fpc_data)

        uname = fpc_data["data"]["attributes"].get("name")
        if uname is None:
            uname = "Anonymous"
        seed = binascii.crc32(uname.encode("utf8"))

        logger.info("Beginning clustering. method=%s inks=%s", method, len(inks_df))
        clustering_t = start_stopwatch()

        if method == "kmeans":
            inks_df, categories = kmeans(
                inks_df=inks_df,
                clusters=request.args.get("clusters", None),
                max_clusters=request.args.get("max_clusters", None),
                delta_e_max=request.args.get("delta_e_max", None),
                random_state=seed,
            )
        elif method == "hdbscan":
            inks_df, categories = hdbscan(
                min_size=request.args.get("min_size", None),
                max_size=request.args.get("max_size", None),
                inks_df=inks_df,
            )
        elif method == "dbscan":
            inks_df, categories = dbscan(
                delta_e_sim=request.args.get("delta_e_sim", None),
                min_size=request.args.get("min_size", None),
                inks_df=inks_df,
            )
        if method == "spectral":
            inks_df, categories = spectral(
                inks_df=inks_df,
                clusters=request.args.get("clusters", None),
                max_clusters=request.args.get("max_clusters", None),
                score_max=request.args.get("score_max", None),
                random_state=seed,
            )
        if method == "agglom":
            inks_df, categories = glom(
                inks_df=inks_df,
                clusters=request.args.get("clusters", None),
                max_clusters=request.args.get("max_clusters", None),
                score_max=request.args.get("score_max", None),
            )

        clustering_t = stop_stopwatch(clustering_t)
        logger.info("Done clustering. method=%s duration=%s", method, clustering_t)

        logger.info("Gathering inks. categories=%s inks=%s", len(categories), len(inks_df))
        gather_t = start_stopwatch()

        in_colors = defaultdict(list)

        for _, category in categories.iterrows():
            inks_in = inks_df[inks_df["cluster"] == category.name].copy()
            inks_in[["L_", "C_", "H_"]] = lab2lch(inks_in[["L", "a", "b"]].values)
            inks_in = inks_in.sort_values("H_")
            for _, ink in inks_in.iterrows():
                in_colors[category.name].append(
                    {"label": ink.label, "color": ink.color, "L": ink.L}
                )

        gather_t = stop_stopwatch(gather_t)
        logger.info("Done gathering inks. categories=%s inks=%s duration=%s", len(categories), len(inks_df), gather_t)

        logger.info("Rendering template. categories=%s inks=%s", len(categories), len(inks_df))
        render_t = start_stopwatch()

        html = render_template(
            "color_pallate.html",
            name=fpc_data["data"]["attributes"]["name"],
            categories=categories.reset_index().to_dict(orient="records"),
            inks=in_colors,
            category_count=len(categories),
        )

        render_t = stop_stopwatch(render_t)
        logger.info("Done rendering template. categories=%s inks=%s duration=%s", len(categories), len(inks_df), render_t)

        return html


    @app.route("/tsne/<int:user_id>")
    def tsne(user_id: int):
        """Generate a TSNE projection of the user's color dataset

        Args:
            user_id (int): The user ID to analyze
        """
        try:
            fpc_data = get_fpc_data(user_id)
        except JSONDecodeError:
            return make_response("Not Found", 404)

        inks_df = fpc_to_df(fpc_data)
        n = len(inks_df)
        lab = inks_df[["L", "a", "b"]].values

        dists = np.zeros((n, n))
        for i in range(n):
            dists[i] = deltaE_ciede2000(lab[i][None, :], lab)

        tsne = TSNE(
            n_components=2,
            random_state=hash(fpc_data["data"]["attributes"]["name"]) % 4294967295,
            metric="precomputed",
            init="random",
        )
        embedding = tsne.fit_transform(dists)

        lab_array = inks_df[["L", "a", "b"]].values.reshape(1, -1, 3)
        rgb_array = lab2rgb(lab_array)[0]
        rgb_hex = [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            for r, g, b in rgb_array
        ]

        plot_df = pd.DataFrame(embedding, columns=["x", "y"])
        plot_df["color"] = rgb_hex

        dpi = 100
        size_in_inches = 2048 / dpi

        fig = plt.figure(figsize=(size_in_inches, size_in_inches), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], color=rgb_array, s=640)
        ax.axis("off")

        buf = BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", transparent=True, facecolor=("#000", 0)
        )
        plt.close()
        buf.seek(0)

        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=False,
            download_name="tsne_plot.png",
        )


    def is_within_delta_e(row: pd.Series, target: np.ndarray, delta_max: float) -> bool:
        """Create a dataframe mask with deltaE_ciede2000

        Args:
            row (pd.Series): The row to check
            target (np.ndarray): The Lab target
            delta_max (float): The maximum color distance

        Returns:
            bool: If the row is within the delta E
        """
        sample = np.array([[row["L"], row["a"], row["b"]]])
        target = target[np.newaxis, :]
        delta_e = deltaE_ciede2000(sample, target)[0]
        return delta_e <= delta_max


    @app.route("/taupe-me/<int:user_id>")
    def taupe(user_id: int):
        """Search for colors with a given term and apply them as categories

        Args:
            user_id (int): The user ID to analyze
        """
        try:
            fpc_data = get_fpc_data(user_id)
        except JSONDecodeError:
            return make_response("Not Found", 404)

        delta_e_max = request.args.get("delta_e_max", 5.0)
        delta_e_max = float(delta_e_max)

        color_name = request.args.get("color_name", "taupe")

        colors = get_named_colors()
        colors = colors[colors["color_name"].str.contains(color_name, case=False)]

        categories = colors[["L_", "a_", "b_", "color_name"]]
        categories = categories.rename(
            columns={"L_": "L", "a_": "a", "b_": "b", "color_name": "name"}
        )
        categories["cluster"] = categories.index
        categories[["L_", "C_", "H_"]] = lab2lch(categories[["L", "a", "b"]].values)
        categories = categories.sort_values("H_")

        inks_df = fpc_to_df(fpc_data)

        in_colors = defaultdict(list)

        for _, category in categories.iterrows():
            mask = inks_df.apply(
                is_within_delta_e,
                axis=1,
                args=(category[["L", "a", "b"]].values, delta_e_max),
            )
            inks_in = inks_df[mask].reset_index(drop=True)
            inks_in["delta_e"] = inks_in.apply(
                lambda row: deltaE_ciede2000(
                    row[["L", "a", "b"]].values, category[["L", "a", "b"]].values
                ),
                axis=1,
            )
            inks_in[["L_", "C_", "H_"]] = lab2lch(inks_in[["L", "a", "b"]].values)
            inks_in = inks_in.sort_values("delta_e")
            for _, ink in inks_in.iterrows():
                in_colors[category.name].append(
                    {"label": ink.label, "color": ink.color, "L": ink.L}
                )

        return render_template(
            "color_pallate.html",
            name=fpc_data["data"]["attributes"]["name"],
            categories=categories.reset_index().to_dict(orient="records"),
            inks=in_colors,
            category_count=len(colors),
        )


    @app.route("/bens-colors/<int:user_id>")
    def bens_colors(user_id: int):
        """Apply my custom categories

        Args:
            user_id (int): The user ID to analyze
        """
        try:
            fpc_data = get_fpc_data(user_id)
        except JSONDecodeError:
            return make_response("Not Found", 404)

        colors = get_named_colors("my_colors.csv", False)

        categories = colors[["L_", "a_", "b_", "color_name"]]
        categories = categories.rename(
            columns={"L_": "L", "a_": "a", "b_": "b", "color_name": "name"}
        )
        categories["cluster"] = categories.index

        inks_df = fpc_to_df(fpc_data)

        ink_lab = inks_df[["L", "a", "b"]].values
        category_lab = categories[["L", "a", "b"]].values

        assigned = []
        for ink in ink_lab:
            de = deltaE_ciede2000(
                ink.reshape(1, 1, 3), category_lab.reshape(-1, 1, 3)
            ).flatten()
            assigned.append(categories.iloc[de.argmin()]["name"])

        inks_df["category"] = assigned

        in_colors = defaultdict(list)

        for _, category in categories.iterrows():
            inks_in = inks_df[inks_df["category"] == category["name"]].reset_index(
                drop=True
            )
            inks_in["delta_e"] = inks_in.apply(
                lambda row: deltaE_ciede2000(
                    row[["L", "a", "b"]].values, category[["L", "a", "b"]].values
                ),
                axis=1,
            )
            inks_in[["L_", "C_", "H_"]] = lab2lch(inks_in[["L", "a", "b"]].values)
            inks_in = inks_in.sort_values("delta_e")
            for _, ink in inks_in.iterrows():
                in_colors[category.name].append(
                    {"label": ink.label, "color": ink.color, "L": ink.L}
                )

        return render_template(
            "color_pallate.html",
            name=fpc_data["data"]["attributes"]["name"],
            categories=categories.reset_index().to_dict(orient="records"),
            inks=in_colors,
            category_count=len(colors),
        )
    
    return app
    
# app = create_app()
