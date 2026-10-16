"""MovieLens download, loading, preprocessing, and deterministic splitting."""
from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def ensure_dataset(data_dir: Path) -> Path:
    """Download and unpack MovieLens 100K when it is not already present."""
    target = data_dir / "ml-100k"
    if (target / "u.data").exists() and (target / "u.item").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ml-100k.zip"
    print("Downloading MovieLens 100K…")
    try:
        import certifi
    except ImportError as error:
        raise RuntimeError(
            "The secure download needs certifi. Run: pip install certifi"
        ) from error
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(DATA_URL, context=context, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception:
        archive.unlink(missing_ok=True)
        raise
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    archive.unlink(missing_ok=True)
    return target


def load_ratings(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movie metadata from MovieLens 100K."""
    root = ensure_dataset(data_dir)
    ratings = pd.read_csv(root / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    movies = pd.read_csv(root / "u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["item_id", "title"])
    return ratings, movies


def describe_ratings(ratings: pd.DataFrame) -> dict[str, float]:
    """Calculate dataset quality and density statistics."""
    users, items = ratings.user_id.nunique(), ratings.item_id.nunique()
    by_user, by_item = ratings.groupby("user_id").size(), ratings.groupby("item_id").size()
    stats = {
        "users": users, "items": items, "ratings": len(ratings),
        "sparsity": 1 - len(ratings) / (users * items),
        "missing": int(ratings.isna().sum().sum()), "duplicates": int(ratings.duplicated(["user_id", "item_id"]).sum()),
        "ratings_per_user_mean": float(by_user.mean()), "ratings_per_user_median": float(by_user.median()),
        "ratings_per_item_mean": float(by_item.mean()), "ratings_per_item_median": float(by_item.median()),
    }
    print(f"{users} users, {items} items, {len(ratings)} ratings, sparsity {stats['sparsity']:.2%}")
    print(f"Missing values: {stats['missing']}; duplicate user-item pairs: {stats['duplicates']}")
    return stats


def reindex_ratings(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Map source IDs to compact zero-based integer indices."""
    users = {value: i for i, value in enumerate(sorted(ratings.user_id.unique()))}
    items = {value: i for i, value in enumerate(sorted(ratings.item_id.unique()))}
    indexed = ratings.copy()
    indexed["user_idx"] = indexed.user_id.map(users)
    indexed["item_idx"] = indexed.item_id.map(items)
    return indexed, users, items


def split_ratings(ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a fixed random rating-level train/test split."""
    train, test = train_test_split(ratings, test_size=test_size, random_state=seed, shuffle=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_matrix(train: pd.DataFrame, n_users: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Build dense ratings matrix and an observed-entry mask."""
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    matrix[train.user_idx.to_numpy(), train.item_idx.to_numpy()] = train.rating.to_numpy(dtype=np.float32)
    return matrix, matrix > 0
