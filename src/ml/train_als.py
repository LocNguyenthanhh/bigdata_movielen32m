"""Huấn luyện mô hình Lọc cộng tác Spark ALS (Alternating Least Squares).

Phụ trách: Nguyễn Văn Tâm (Spark MLlib Modeling)
"""

import time
from typing import Tuple
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame

from src.common.config import PROJECT_ROOT, load_config
from src.common.spark_session import get_spark_session


def load_and_split_ratings(env: str = "dev") -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Đọc dữ liệu ratings Parquet, lấy mẫu theo cấu hình và chia tập train/test."""
    cfg = load_config(env)
    spark = get_spark_session(cfg)

    ratings_path = str(PROJECT_ROOT / cfg["paths"]["ratings_parquet"])
    sample_fraction = float(cfg.get("data", {}).get("sample_fraction", 0.05))
    seed = int(cfg["project"]["seed"])

    df = spark.read.parquet(ratings_path)
    if sample_fraction < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)

    train_ratio = float(cfg["evaluation"]["train_ratio"])
    test_ratio = float(cfg["evaluation"]["test_ratio"])
    train_df, test_df = df.randomSplit([train_ratio, test_ratio], seed=seed)

    return df, train_df, test_df


def train_als_model(train_df: DataFrame, env: str = "dev") -> Tuple[ALSModel, float]:
    """Huấn luyện mô hình Spark ALS trên tập train."""
    cfg = load_config(env)
    als_cfg = cfg["als"]
    seed = int(cfg["project"]["seed"])

    als = ALS(
        userCol=als_cfg["user_col"],
        itemCol=als_cfg["item_col"],
        ratingCol=als_cfg["rating_col"],
        rank=int(als_cfg["rank"]),
        maxIter=int(als_cfg["max_iter"]),
        regParam=float(als_cfg["reg_param"]),
        coldStartStrategy=als_cfg["cold_start_strategy"],
        nonnegative=bool(als_cfg["nonnegative"]),
        checkpointInterval=int(als_cfg["checkpoint_interval"]),
        seed=seed,
    )

    start_time = time.perf_counter()
    model = als.fit(train_df)
    elapsed = time.perf_counter() - start_time
    print(f"[ALS] Huấn luyện hoàn tất trong {elapsed:.2f} giây.")
    return model, elapsed


if __name__ == "__main__":
    _, train_data, _ = load_and_split_ratings("dev")
    train_als_model(train_data, "dev")
