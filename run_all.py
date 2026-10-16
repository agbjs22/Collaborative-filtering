"""Run the complete collaborative-filtering recommendation pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_ratings, describe_ratings, reindex_ratings, split_ratings, build_matrix
from src.user_based import UserBasedCF
from src.item_based import ItemBasedCF
from src.svd_model import SVDRecommender
from src.popularity import PopularityRecommender
from src.evaluate import evaluate_model
from src.experiments import save_bar_chart, save_distribution_plots, neighbourhood_sweep, similarity_comparison, sparsity_analysis, cold_start_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Evaluate rankings on 200 users and skip extended retraining experiments")
    args = parser.parse_args()
    root, results = Path(__file__).parent, Path(__file__).parent / "results"
    figures = results / "figures"; figures.mkdir(parents=True, exist_ok=True)
    ratings, movies = load_ratings(root / "data")
    describe_ratings(ratings); indexed, _, item_map = reindex_ratings(ratings)
    train, test = split_ratings(indexed)
    matrix, _ = build_matrix(train, indexed.user_idx.nunique(), indexed.item_idx.nunique())
    models = {
        "User-based CF (cosine)": UserBasedCF("cosine"),
        "User-based CF (Pearson)": UserBasedCF("pearson"),
        "Item-based CF (cosine)": ItemBasedCF("cosine"),
        "Item-based CF (adjusted cosine)": ItemBasedCF("adjusted_cosine"),
        "SVD": SVDRecommender(),
        "Popularity": PopularityRecommender(),
    }
    records, ranking_users = [], 200 if args.quick else None
    for name, model in models.items():
        print(f"Training {name}…")
        model.fit(matrix)
        records.append({"Model": name, **evaluate_model(model, matrix, test, max_ranking_users=ranking_users)})
    metrics = pd.DataFrame(records); metrics.to_csv(results / "metrics.csv", index=False)
    best_item = metrics[metrics.Model.str.startswith("Item-based")].sort_values("RMSE").iloc[0]
    final = pd.DataFrame([metrics.iloc[0], metrics.iloc[1], best_item, metrics[metrics.Model == "SVD"].iloc[0]])
    columns = ["Model", "RMSE", "MAE", "Precision@10", "Recall@10", "NDCG@10"]
    lines = ["| " + " | ".join(columns) + " |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in final.iterrows():
        lines.append("| " + " | ".join([str(row["Model"])] + [f"{float(row[column]):.4f}" for column in columns[1:]]) + " |")
    (results / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_distribution_plots(ratings, figures); save_bar_chart(records, figures / "model_comparison.png"); similarity_comparison(records, figures)
    if args.quick:
        neighbourhood_sweep(matrix, test, figures)
    else:
        neighbourhood_sweep(matrix, test, figures); sparsity_analysis(matrix, test, figures); cold_start_analysis(matrix, test, figures)
    winner = metrics.sort_values("RMSE").iloc[0]
    scope_note = "The quick run omits sparsity and cold-start retraining." if args.quick else "The sparsity and cold-start experiments quantify performance under reduced interaction data."
    (results / "summary.md").write_text(f"{winner.Model} achieved the lowest RMSE of {winner.RMSE:.4f} in this run. The comparison uses the same fixed 80/20 split for every model. User and item neighbourhood models test alternative similarity definitions. {scope_note} Popularity supplies a fallback when collaborative evidence is insufficient. Ranking metrics in metrics.csv report top-10 recommendation quality. The final model choice should consider both rating error and ranking quality.\n", encoding="utf-8")
    best_model = models[winner.Model]
    titles = movies.set_index("item_id").title.to_dict(); inverse_items = {index: item for item, index in item_map.items()}
    sample_users = [0, min(1, matrix.shape[0] - 1), min(2, matrix.shape[0] - 1)]
    rec_lines = ["# Sample recommendations", ""]
    for user in sample_users:
        rated = np.argsort(matrix[user])[::-1]; rated = [item for item in rated if matrix[user, item] > 0][:5]
        scores = best_model.score_all_items(user); scores[matrix[user] > 0] = -float("inf")
        recommended = np.argsort(scores)[::-1][:10]
        rec_lines += [f"## User {user + 1}", "Top-rated training movies: " + ", ".join(titles.get(inverse_items[item], "Unknown") for item in rated), "Recommendations: " + ", ".join(titles.get(inverse_items[item], "Unknown") for item in recommended), ""]
    (results / "sample_recommendations.md").write_text("\n".join(rec_lines), encoding="utf-8")
    print(metrics.to_string(index=False)); print(f"Saved results to {results}")

if __name__ == "__main__": main()
