"""Experiment helpers, hyperparameter sweeps, sparsity & cold-start analysis, and charts."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from .user_based import UserBasedCF
from .item_based import ItemBasedCF
from .svd_model import SVDRecommender
from .popularity import PopularityRecommender
from .evaluate import evaluate_model, rating_metrics


def _test_rmse(model, test: pd.DataFrame, users=None) -> float:
    user_col = "u" if "u" in test.columns else "user_idx"
    item_col = "i" if "i" in test.columns else "item_idx"
    subset = test[test[user_col].isin(users)] if users is not None else test
    truth = subset["rating"].to_numpy(dtype=float)
    predicted = np.array([model.predict(int(u), int(i)) for u, i in zip(subset[user_col], subset[item_col])])
    err = predicted - truth
    return float(np.sqrt(np.mean(err ** 2)))


def save_distribution_plots(ratings: pd.DataFrame, figures: Path) -> None:
    """Save rating distribution and ratings-per-user histograms."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    rating_col = "rating" if "rating" in ratings.columns else "r"
    user_col = "user_id" if "user_id" in ratings.columns else "u"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    counts = ratings[rating_col].value_counts().sort_index()
    ax[0].bar(counts.index, counts.values, color="#1F3864", edgecolor="none")
    for x, v in zip(counts.index, counts.values):
        ax[0].text(x, v + 400, str(v), ha="center", fontsize=9)
    ax[0].set_xlabel("Rating")
    ax[0].set_ylabel("Number of ratings")
    ax[0].set_title("Rating distribution")

    user_counts = ratings.groupby(user_col).size()
    ax[1].hist(user_counts, bins=40, color="#1F3864", edgecolor="white")
    ax[1].set_xlabel("Ratings per user")
    ax[1].set_ylabel("Number of users")
    ax[1].set_title("Ratings per user")

    plt.tight_layout()
    plt.savefig(figures / "rating_distribution.png", dpi=170, facecolor="white")
    plt.savefig(figures / "ratings_per_user.png", dpi=170, facecolor="white")
    plt.close()


def neighbourhood_sweep(matrix: np.ndarray, test: pd.DataFrame, figures: Path) -> pd.DataFrame:
    """Evaluate test RMSE across neighbourhood sizes k in [5, 10, 20, 40, 80] for all 4 CF configurations."""
    ks = [5, 10, 20, 40, 80]
    configs = {
        "User-based (cosine)": lambda k: UserBasedCF("cosine", k=k).fit(matrix),
        "User-based (Pearson)": lambda k: UserBasedCF("pearson", k=k).fit(matrix),
        "Item-based (cosine)": lambda k: ItemBasedCF("cosine", k=k).fit(matrix),
        "Item-based (adjusted cosine)": lambda k: ItemBasedCF("adjusted_cosine", k=k).fit(matrix),
    }

    records = {name: [] for name in configs}
    for k in ks:
        for name, fn in configs.items():
            model = fn(k)
            records[name].append(_test_rmse(model, test))

    k_df = pd.DataFrame(records, index=ks)
    k_df.index.name = "k"

    # Line chart of RMSE vs k
    plt.figure(figsize=(8, 4.8))
    markers = {"User-based (cosine)": "o", "User-based (Pearson)": "s", "Item-based (cosine)": "^", "Item-based (adjusted cosine)": "D"}
    colors = {"User-based (cosine)": "#1F3864", "User-based (Pearson)": "#6F8FC9", "Item-based (cosine)": "#B5651D", "Item-based (adjusted cosine)": "#E0A96D"}
    for name in k_df.columns:
        plt.plot(ks, k_df[name], marker=markers[name], color=colors[name], label=name, lw=2)
    plt.xscale("log")
    plt.xticks(ks, ks)
    plt.xlabel("Neighbourhood size k")
    plt.ylabel("Test RMSE")
    plt.title("RMSE versus neighbourhood size")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "rmse_vs_k.png", dpi=170, facecolor="white")
    plt.close()

    return k_df


def sparsity_analysis(matrix: np.ndarray, test: pd.DataFrame, train_df: pd.DataFrame, figures: Path) -> pd.DataFrame:
    """Evaluate how performance degrades as training data fraction drops from 100% to 20%."""
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    u_col = "u" if "u" in train_df.columns else "user_idx"
    i_col = "i" if "i" in train_df.columns else "item_idx"
    r_col = "rating"

    rows = []
    for f in fractions:
        sub = train_df.sample(frac=f, random_state=42)
        rf = np.zeros_like(matrix)
        rf[sub[u_col].to_numpy(), sub[i_col].to_numpy()] = sub[r_col].to_numpy()
        sparsity_pct = 100.0 * (1.0 - np.count_nonzero(rf) / rf.size)

        ub = UserBasedCF("pearson", k=40).fit(rf)
        ib = ItemBasedCF("adjusted_cosine", k=40).fit(rf)
        svd = SVDRecommender(n_factors=50, n_epochs=30, lr=0.005, reg=0.02, seed=42).fit(rf)

        rows.append({
            "fraction": f,
            "sparsity %": sparsity_pct,
            "User-based (Pearson)": _test_rmse(ub, test),
            "Item-based (adjusted cosine)": _test_rmse(ib, test),
            "SVD": _test_rmse(svd, test),
        })

    sp_df = pd.DataFrame(rows).set_index("fraction")

    # Plot
    plt.figure(figsize=(8, 4.8))
    plt.plot(sp_df["sparsity %"], sp_df["User-based (Pearson)"], marker="o", color="#6F8FC9", label="User-based (Pearson)", lw=2)
    plt.plot(sp_df["sparsity %"], sp_df["Item-based (adjusted cosine)"], marker="o", color="#E0A96D", label="Item-based (adjusted cosine)", lw=2)
    plt.plot(sp_df["sparsity %"], sp_df["SVD"], marker="o", color="#2E7D32", label="SVD", lw=2)
    plt.xlabel("Sparsity of training matrix (%)")
    plt.ylabel("Test RMSE")
    plt.title("RMSE versus data sparsity")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "rmse_vs_sparsity.png", dpi=170, facecolor="white")
    plt.close()

    return sp_df


def cold_start_analysis(matrix: np.ndarray, test: pd.DataFrame, figures: Path) -> pd.DataFrame:
    """Compare CF, Popularity fallback, and a Hybrid rule for cold-start users (1 to 20 ratings kept)."""
    user_col = "u" if "u" in test.columns else "user_idx"
    test_by_user = {u: g for u, g in test.groupby(user_col)}
    mask = matrix > 0
    n_users = matrix.shape[0]

    rng = np.random.RandomState(42)
    eligible = [u for u, g in test_by_user.items() if (g["rating"] >= 4).any() and mask[u].sum() >= 20]
    cold_users = rng.choice(eligible, size=int(0.1 * n_users), replace=False)

    cold_rows = []
    for n in [1, 3, 5, 10, 20]:
        r2, m2 = matrix.copy(), mask.copy()
        for u in cold_users:
            rated = np.where(mask[u])[0]
            drop = np.setdiff1d(rated, rng.choice(rated, n, replace=False))
            r2[u, drop] = 0.0
            m2[u, drop] = False

        cf_model = ItemBasedCF("adjusted_cosine", k=40).fit(r2)
        pop_model = PopularityRecommender().fit(r2)

        cf_eval = evaluate_model(cf_model, r2, test[test[user_col].isin(cold_users)], k=10)
        pop_eval = evaluate_model(pop_model, r2, test[test[user_col].isin(cold_users)], k=10, rated_scores=False)

        # Baseline item-mean RMSE
        counts = m2.sum(axis=0)
        global_m = r2[m2].mean() if m2.any() else 3.0
        im = np.where(counts > 0, r2.sum(axis=0) / np.maximum(counts, 1), global_m)
        subset = test[test[user_col].isin(cold_users)]
        base_err = subset["rating"].to_numpy(dtype=float) - im[subset["i" if "i" in subset.columns else "item_idx"].to_numpy(dtype=int)]
        base_rmse = float(np.sqrt(np.mean(base_err ** 2)))

        hybrid_p10 = pop_eval["Precision@10"] if n < 5 else cf_eval["Precision@10"]
        hybrid_rmse = base_rmse if n < 5 else cf_eval["RMSE"]

        cold_rows.append({
            "ratings kept": n,
            "CF P@10": cf_eval["Precision@10"],
            "Popularity P@10": pop_eval["Precision@10"],
            "Hybrid P@10": hybrid_p10,
            "CF RMSE": cf_eval["RMSE"],
            "Item-mean RMSE": base_rmse,
            "Hybrid RMSE": hybrid_rmse,
        })

    cold_df = pd.DataFrame(cold_rows).set_index("ratings kept")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(cold_df.index, cold_df["CF P@10"], marker="o", label="CF (item-based, adj. cosine)", lw=2)
    ax[0].plot(cold_df.index, cold_df["Popularity P@10"], marker="s", label="Popularity", lw=2)
    ax[0].plot(cold_df.index, cold_df["Hybrid P@10"], marker="^", label="Hybrid (popularity if < 5)", lw=2)
    ax[0].set_xlabel("Ratings kept per cold-start user")
    ax[0].set_ylabel("Precision@10")
    ax[0].set_title("Cold start: Precision@10")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.25)

    ax[1].plot(cold_df.index, cold_df["CF RMSE"], marker="o", label="CF", lw=2)
    ax[1].plot(cold_df.index, cold_df["Item-mean RMSE"], marker="s", label="Item-mean fallback", lw=2)
    ax[1].plot(cold_df.index, cold_df["Hybrid RMSE"], marker="^", label="Hybrid", lw=2)
    ax[1].set_xlabel("Ratings kept per cold-start user")
    ax[1].set_ylabel("Test RMSE")
    ax[1].set_title("Cold start: RMSE")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(figures / "cold_start.png", dpi=170, facecolor="white")
    plt.close()

    return cold_df


def save_standalone_visualizations(res_df: pd.DataFrame, figures: Path) -> None:
    """Save dedicated visualizations for RMSE/MAE, Precision/Recall/F1, and Model Ranking."""
    models_with_rmse = res_df.dropna(subset=["RMSE"])
    labels = [m.replace("CF (", "\n(").replace("baseline", "\n(baseline)") for m in models_with_rmse.index]
    x = np.arange(len(models_with_rmse))
    width = 0.35

    # 1. RMSE and MAE Comparison
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width/2, models_with_rmse["RMSE"], width, label="RMSE", color="#1F3864")
    ax.bar(x + width/2, models_with_rmse["MAE"], width, label="MAE", color="#6F8FC9")
    for i in x:
        ax.text(i - width/2, models_with_rmse["RMSE"].iloc[i] + 0.01, f"{models_with_rmse['RMSE'].iloc[i]:.4f}", ha="center", fontsize=8)
        ax.text(i + width/2, models_with_rmse["MAE"].iloc[i] + 0.01, f"{models_with_rmse['MAE'].iloc[i]:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Error")
    ax.set_ylim(0, 1.15)
    ax.set_title("Rating Prediction Error Comparison (RMSE vs MAE)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "rmse_mae_comparison.png", dpi=170, facecolor="white")
    plt.close()

    # 2. Precision@10, Recall@10, and F1@10 Comparison
    fig, ax = plt.subplots(figsize=(10, 4.8))
    all_labels = [m.replace("CF (", "\n(").replace("baseline", "\n(baseline)") for m in res_df.index]
    all_x = np.arange(len(res_df))
    w = 0.25
    ax.bar(all_x - w, res_df["Precision@10"], w, label="Precision@10", color="#1F3864")
    ax.bar(all_x, res_df["Recall@10"], w, label="Recall@10", color="#6F8FC9")
    f1_col = "F1@10" if "F1@10" in res_df.columns else res_df["Precision@10"]
    ax.bar(all_x + w, f1_col, w, label="F1@10", color="#E0A96D")
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Top-10 Recommendation Quality (Precision, Recall, F1)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "precision_recall_f1_comparison.png", dpi=170, facecolor="white")
    plt.close()


def save_publication_composite_figures(
    res_df: pd.DataFrame,
    k_df: pd.DataFrame,
    sp_df: pd.DataFrame,
    cold_df: pd.DataFrame,
    figures: Path,
) -> None:
    """Generate the 2 exact composite multi-panel figures that match the DMT Project Report (Fig 11.4.6 & 11.4.7)."""
    col = {
        "User-based (cosine)": "#1F3864",
        "User-based (Pearson)": "#6F8FC9",
        "Item-based (cosine)": "#B5651D",
        "Item-based (adjusted cosine)": "#E0A96D",
        "SVD": "#2E7D32",
    }
    mk = {
        "User-based (cosine)": "o",
        "User-based (Pearson)": "s",
        "Item-based (cosine)": "^",
        "Item-based (adjusted cosine)": "D",
    }

    # Fig 11.4.6 (2x2 Panel)
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.2))
    ax = ax.ravel()
    for c in k_df.columns:
        ax[0].plot(k_df.index, k_df[c], marker=mk[c], color=col[c], label=c, lw=2)
    ax[0].set_xscale("log")
    ax[0].set_xticks(k_df.index)
    ax[0].set_xticklabels(k_df.index)
    ax[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[0].set_xlabel("Neighbourhood size k")
    ax[0].set_ylabel("Test RMSE")
    ax[0].set_title("(a) RMSE versus neighbourhood size")
    ax[0].legend(fontsize=9, frameon=False)
    ax[0].grid(alpha=0.25)

    best = k_df.min()
    lab = ["UB\ncosine", "UB\nPearson", "IB\ncosine", "IB adj.\ncosine"]
    bb = ax[1].bar(lab, best.values, color=[col[c] for c in k_df.columns])
    for r, v, kk in zip(bb, best.values, k_df.idxmin().values):
        ax[1].text(r.get_x() + r.get_width() / 2, v + 0.0015, f"{v:.4f}\n(k={kk})", ha="center", fontsize=9)
    ax[1].set_ylim(best.min() - 0.03, best.max() + 0.02)
    ax[1].set_ylabel("Test RMSE at best k (axis truncated)")
    ax[1].set_title("(b) Similarity measure comparison")

    # (c) RMSE of all models
    res_rmse = res_df["RMSE"].dropna()
    ml = ["UB\ncosine", "UB\nPearson", "IB\ncosine", "IB adj.\ncosine", "SVD"]
    col_list = [col["User-based (cosine)"], col["User-based (Pearson)"], col["Item-based (cosine)"], col["Item-based (adjusted cosine)"], col["SVD"]]
    bb = ax[2].bar(ml, res_rmse.values[:5], color=col_list)
    for r, v in zip(bb, res_rmse.values[:5]):
        ax[2].text(r.get_x() + r.get_width() / 2, v + 0.0015, f"{v:.4f}", ha="center", fontsize=9)
    ax[2].set_ylim(res_rmse.min() - 0.03, res_rmse.max() + 0.02)
    ax[2].set_ylabel("Test RMSE, k = 40 (axis truncated)")
    ax[2].set_title("(c) RMSE of all models")

    # (d) Top-10 recommendation quality
    short = ["UB cos", "UB Pear", "IB cos", "IB adj", "SVD", "Popular"]
    x = np.arange(len(res_df))
    wdt = 0.27
    for j, (m, cl) in enumerate([("Precision@10", "#1F3864"), ("Recall@10", "#6F8FC9"), ("NDCG@10", "#E0A96D")]):
        ax[3].bar(x + (j - 1) * wdt, res_df[m], wdt, label=m, color=cl)
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(short, fontsize=9)
    ax[3].set_ylabel("Score")
    ax[3].set_title("(d) Top-10 recommendation quality")
    ax[3].legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(figures / "fig_11_4_6_k_similarity_graphs.png", dpi=170, facecolor="white")
    plt.savefig(figures / "similarity_comparison.png", dpi=170, facecolor="white")
    plt.close()

    # Fig 11.4.7 (Sparsity + Cold-start 3-panel)
    fig = plt.figure(figsize=(10.5, 8.2))
    gs = fig.add_gridspec(2, 4)
    a0 = fig.add_subplot(gs[0, 0:2])
    a1 = fig.add_subplot(gs[0, 2:4])
    a2 = fig.add_subplot(gs[1, 1:3])

    for c in ["User-based (Pearson)", "Item-based (adjusted cosine)", "SVD"]:
        a0.plot(sp_df["sparsity %"], sp_df[c], marker="o", color=col[c], label=c, lw=2)
    a0.set_xlabel("Sparsity of training matrix (%)")
    a0.set_ylabel("Test RMSE")
    a0.set_title("(a) RMSE versus data sparsity")
    a0.legend(frameon=False, fontsize=9)
    a0.grid(alpha=0.25)

    for c, l, m in [("CF P@10", "CF (item-based, adj. cosine)", "o"), ("Popularity P@10", "Popularity", "s"), ("Hybrid P@10", "Hybrid (popularity if < 5)", "^")]:
        a1.plot(cold_df.index, cold_df[c], marker=m, lw=2, label=l)
    a1.set_xlabel("Ratings kept per cold-start user")
    a1.set_ylabel("Precision@10")
    a1.set_title("(b) Cold start: Precision@10")
    a1.set_xticks(cold_df.index)
    a1.legend(frameon=False, fontsize=9, loc="center right")
    a1.grid(alpha=0.25)

    for c, l, m in [("CF RMSE", "CF", "o"), ("Item-mean RMSE", "Item-mean fallback", "s"), ("Hybrid RMSE", "Hybrid", "^")]:
        a2.plot(cold_df.index, cold_df[c], marker=m, lw=2, label=l)
    a2.set_xlabel("Ratings kept per cold-start user")
    a2.set_ylabel("Test RMSE")
    a2.set_title("(c) Cold start: RMSE")
    a2.set_xticks(cold_df.index)
    a2.legend(frameon=False, fontsize=9)
    a2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(figures / "fig_11_4_7_sparsity_coldstart_graphs.png", dpi=170, facecolor="white")
    plt.savefig(figures / "model_comparison.png", dpi=170, facecolor="white")
    plt.close()

    # Also generate the standalone specific visualizations requested
    save_standalone_visualizations(res_df, figures)
