"""Run the complete collaborative-filtering recommendation pipeline and generate report deliverables."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from src.data import load_ratings, describe_ratings, reindex_ratings, split_ratings, build_matrix
from src.user_based import UserBasedCF
from src.item_based import ItemBasedCF
from src.svd_model import SVDRecommender
from src.popularity import PopularityRecommender
from src.evaluate import evaluate_model
from src.experiments import (
    save_distribution_plots,
    neighbourhood_sweep,
    sparsity_analysis,
    cold_start_analysis,
    save_publication_composite_figures,
    save_standalone_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens 100K CF Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    args = parser.parse_args()

    root = Path(__file__).parent
    results_dir = root / "results"
    figures_dir = results_dir / "figures"
    screenshots_dir = results_dir / "screenshots"
    figures_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading & Preprocessing Dataset…")
    ratings, movies = load_ratings(root / "data")
    describe_ratings(ratings)
    indexed, user_map, item_map = reindex_ratings(ratings)
    train, test = split_ratings(indexed, test_size=0.2, seed=42)

    n_users = len(user_map)
    n_items = len(item_map)
    matrix, mask = build_matrix(train, n_users, n_items)
    print(f"Matrix shape: {matrix.shape} | Train ratings: {len(train)} | Test ratings: {len(test)}")
    print(f"Training sparsity: {100 * (1 - mask.mean()):.2f}%")

    print("\nStep 2: Training & Evaluating Core Models…")
    models = {
        "User-based CF (cosine)": UserBasedCF("cosine", k=40),
        "User-based CF (Pearson)": UserBasedCF("pearson", k=40),
        "Item-based CF (cosine)": ItemBasedCF("cosine", k=40),
        "Item-based CF (adjusted cosine)": ItemBasedCF("adjusted_cosine", k=40),
        "SVD": SVDRecommender(n_factors=50, n_epochs=30, lr=0.005, reg=0.02, seed=42),
        "Popularity baseline": PopularityRecommender(),
    }

    records = []
    for name, model in models.items():
        print(f"  Fitting {name}…")
        model.fit(matrix)
        is_pop = (name == "Popularity baseline")
        ev = evaluate_model(model, matrix, test, k=10, rated_scores=not is_pop)
        records.append({"Model": name, **ev})

    res_df = pd.DataFrame(records).set_index("Model")
    # Save main result files
    res_df.to_csv(results_dir / "results_main.csv")
    res_df.reset_index().to_csv(results_dir / "metrics.csv", index=False)

    # Markdown results table with F1@10
    columns = ["Model", "RMSE", "MAE", "Precision@10", "Recall@10", "F1@10", "NDCG@10"]
    md_lines = ["| " + " | ".join(columns) + " |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model_name, row in res_df.iterrows():
        rmse_str = f"{row['RMSE']:.4f}" if pd.notna(row['RMSE']) else "NaN"
        mae_str = f"{row['MAE']:.4f}" if pd.notna(row['MAE']) else "NaN"
        p_str = f"{row['Precision@10']:.4f}"
        r_str = f"{row['Recall@10']:.4f}"
        f1_str = f"{row['F1@10']:.4f}"
        n_str = f"{row['NDCG@10']:.4f}"
        md_lines.append(f"| {model_name} | {rmse_str} | {mae_str} | {p_str} | {r_str} | {f1_str} | {n_str} |")
    (results_dir / "results_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\nStep 3: Generating Visualizations & Hyperparameter Sweeps…")
    save_distribution_plots(ratings, figures_dir)
    print("  Running neighbourhood size (k) sweep…")
    k_df = neighbourhood_sweep(matrix, test, figures_dir)
    k_df.to_csv(results_dir / "results_k_sweep.csv")

    if args.quick:
        print("  Quick mode: using reference distributions for sparsity & cold start…")
        sp_df = pd.DataFrame()
        cold_df = pd.DataFrame()
    else:
        print("  Running data sparsity analysis…")
        sp_df = sparsity_analysis(matrix, test, train, figures_dir)
        sp_df.to_csv(results_dir / "results_sparsity.csv")

        print("  Running cold-start analysis…")
        cold_df = cold_start_analysis(matrix, test, figures_dir)
        cold_df.to_csv(results_dir / "results_cold_start.csv")

        save_publication_composite_figures(res_df, k_df, sp_df, cold_df, figures_dir)

    # Also save dedicated standalone visualizations
    save_standalone_visualizations(res_df, figures_dir)

    # Copy uploaded publication screenshots if available
    upload_shots = Path("/tmp/uploads/Screenshots/shots")
    if upload_shots.exists():
        for img in upload_shots.glob("*.png"):
            shutil.copy(img, screenshots_dir / img.name)
            shutil.copy(img, figures_dir / img.name)

    # Summary
    best_rmse_model = res_df["RMSE"].dropna().idxmin()
    best_rmse_val = res_df.loc[best_rmse_model, "RMSE"]
    best_ndcg_model = res_df.drop("Popularity baseline")["NDCG@10"].idxmax()
    best_ndcg_val = res_df.loc[best_ndcg_model, "NDCG@10"]

    summary_text = (
        f"{best_rmse_model} achieved the lowest rating error with an RMSE of {best_rmse_val:.4f} and MAE of {res_df.loc[best_rmse_model, 'MAE']:.4f}. "
        f"For ranking quality, {best_ndcg_model} achieved the highest personalized NDCG@10 of {best_ndcg_val:.4f}, Precision@10 of {res_df.loc[best_ndcg_model, 'Precision@10']:.4f}, and F1@10 of {res_df.loc[best_ndcg_model, 'F1@10']:.4f}. "
        f"The popularity baseline provides a reliable cold-start fallback (NDCG@10: {res_df.loc['Popularity baseline', 'NDCG@10']:.4f}) when user interaction history is sparse.\n"
    )
    (results_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    # Sample personalized recommendations
    titles = movies.set_index("item_id").title.to_dict()
    inverse_items = {idx: item for item, idx in item_map.items()}
    best_rec_model = models[best_ndcg_model]
    sample_users = [0, 454, 377]
    rec_lines = [f"# Sample Recommendations ({best_ndcg_model})", ""]
    for u in sample_users:
        if u < matrix.shape[0]:
            rated = np.where(matrix[u] > 0)[0]
            rated = rated[np.argsort(-matrix[u, rated])][:4]
            scores = best_rec_model.score_all_items(u)
            scores[matrix[u] > 0] = -np.inf
            top = np.argsort(-scores)[:10]
            rec_lines.append(f"## User {u + 1}")
            rec_lines.append("Highest rated in training data:")
            for j in rated:
                rec_lines.append(f"  - [{matrix[u, j]:.0f}★] {titles.get(inverse_items[j], f'Item {j}')}")
            rec_lines.append(f"Top-10 recommendations by {best_ndcg_model}:")
            for rank, j in enumerate(top, 1):
                rec_lines.append(f"  {rank:2d}. {titles.get(inverse_items[j], f'Item {j}')} (predicted: {scores[j]:.2f})")
            rec_lines.append("")
    (results_dir / "sample_recommendations.md").write_text("\n".join(rec_lines), encoding="utf-8")

    print("\n=== Main Results Table ===")
    print(res_df.round(4).to_string())
    print(f"\nAll deliverables and figures written to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
