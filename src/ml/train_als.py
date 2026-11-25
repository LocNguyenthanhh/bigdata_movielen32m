"""Huấn luyện mô hình Lọc cộng tác Spark ALS (Alternating Least Squares).

Gồm:
    - Đọc ratings Parquet, lấy mẫu theo cấu hình và chia train/test.
    - Huấn luyện ALS kèm Checkpointing (cắt DAG Lineage, tránh StackOverflowError).
    - Tinh chỉnh siêu tham số (grid `rank` x `regParam`) trên tập validation tách từ train.
    - Lưu / nạp mô hình ALS (dùng cho Top-N, demo và phân cụm K-Means).

Phụ trách: Nguyễn Văn Tâm (Spark MLlib Modeling)

Chạy độc lập:
    python -m src.ml.train_als --env dev
    python -m src.ml.train_als --env dev --tune
"""

import argparse
import csv
import time
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.evaluation import RegressionEvaluator
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

    df = spark.read.parquet(ratings_path).select("user_id", "movie_id", "rating", "timestamp")
    if sample_fraction < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)

    train_ratio = float(cfg["evaluation"]["train_ratio"])
    test_ratio = float(cfg["evaluation"]["test_ratio"])
    train_df, test_df = df.randomSplit([train_ratio, test_ratio], seed=seed)

    return df, train_df, test_df


def build_als(cfg: Dict[str, Any], rank: Optional[int] = None, reg_param: Optional[float] = None,
              max_iter: Optional[int] = None) -> ALS:
    """Tạo estimator ALS theo cấu hình (có thể ghi đè rank / regParam / maxIter)."""
    als_cfg = cfg["als"]
    return ALS(
        userCol=als_cfg["user_col"],
        itemCol=als_cfg["item_col"],
        ratingCol=als_cfg["rating_col"],
        rank=int(rank if rank is not None else als_cfg["rank"]),
        maxIter=int(max_iter if max_iter is not None else als_cfg["max_iter"]),
        regParam=float(reg_param if reg_param is not None else als_cfg["reg_param"]),
        coldStartStrategy=als_cfg["cold_start_strategy"],
        nonnegative=bool(als_cfg["nonnegative"]),
        checkpointInterval=int(als_cfg["checkpoint_interval"]),
        seed=int(cfg["project"]["seed"]),
    )


def train_als_model(train_df: DataFrame, env: str = "dev", **overrides: Any) -> Tuple[ALSModel, float]:
    """Huấn luyện mô hình Spark ALS trên tập train.

    Args:
        train_df: DataFrame (user_id, movie_id, rating).
        env: Môi trường cấu hình.
        overrides: Ghi đè `rank`, `reg_param`, `max_iter` (dùng khi tuning).

    Returns:
        (Mô hình ALS đã huấn luyện, thời gian huấn luyện tính bằng giây).
    """
    cfg = load_config(env)
    als = build_als(cfg, **overrides)

    start_time = time.perf_counter()
    model = als.fit(train_df)
    elapsed = time.perf_counter() - start_time
    print(
        f"[ALS] rank={als.getRank()} regParam={als.getRegParam()} maxIter={als.getMaxIter()} "
        f"| Huấn luyện hoàn tất trong {elapsed:.2f} giây."
    )
    return model, elapsed


def tune_als_hyperparameters(train_df: DataFrame, env: str = "dev") -> Dict[str, Any]:
    """Grid search `rank` x `regParam`, đánh giá RMSE trên tập validation tách từ train.

    Không dùng tập test để chọn siêu tham số (tránh rò rỉ dữ liệu). Kết quả từng cấu hình
    được ghi vào `paths.tuning_csv` để đưa vào báo cáo.

    Returns:
        Dictionary cấu hình tốt nhất: {"rank", "reg_param", "rmse", "train_time_sec"}.
    """
    cfg = load_config(env)
    tuning_cfg = cfg.get("tuning", {})
    ranks: List[int] = [int(r) for r in tuning_cfg.get("ranks", [10, 20, 50])]
    reg_params: List[float] = [float(r) for r in tuning_cfg.get("reg_params", [0.01, 0.1])]
    val_ratio = float(tuning_cfg.get("validation_ratio", 0.1))
    seed = int(cfg["project"]["seed"])

    fit_df, val_df = train_df.randomSplit([1.0 - val_ratio, val_ratio], seed=seed)
    fit_df = fit_df.cache()
    val_df = val_df.cache()

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol=cfg["als"]["rating_col"],
        predictionCol="prediction",
    )

    results: List[Dict[str, Any]] = []
    for rank in ranks:
        for reg_param in reg_params:
            model, elapsed = train_als_model(fit_df, env=env, rank=rank, reg_param=reg_param)
            rmse = float(evaluator.evaluate(model.transform(val_df)))
            results.append({
                "environment": env,
                "rank": rank,
                "reg_param": reg_param,
                "max_iter": int(cfg["als"]["max_iter"]),
                "val_rmse": round(rmse, 4),
                "train_time_sec": round(elapsed, 2),
            })
            print(f"[TUNING] rank={rank:<3} regParam={reg_param:<5} -> val RMSE={rmse:.4f}")

    fit_df.unpersist()
    val_df.unpersist()

    tuning_path = PROJECT_ROOT / cfg["paths"]["tuning_csv"]
    tuning_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tuning_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = min(results, key=lambda r: r["val_rmse"])
    print(
        f"[TUNING] Tốt nhất: rank={best['rank']} regParam={best['reg_param']} "
        f"(val RMSE={best['val_rmse']}) | Bảng kết quả: {tuning_path}"
    )
    return {
        "rank": best["rank"],
        "reg_param": best["reg_param"],
        "rmse": best["val_rmse"],
        "train_time_sec": best["train_time_sec"],
    }


def save_als_model(model: ALSModel, env: str = "dev") -> str:
    """Lưu mô hình ALS ra `paths.als_model` (ghi đè)."""
    cfg = load_config(env)
    model_path = str(PROJECT_ROOT / cfg["paths"]["als_model"])
    model.write().overwrite().save(model_path)
    print(f"[ALS] Đã lưu mô hình tại {model_path}")
    return model_path


def load_als_model(env: str = "dev") -> ALSModel:
    """Nạp mô hình ALS đã lưu (cần SparkSession đang chạy)."""
    cfg = load_config(env)
    get_spark_session(cfg)
    return ALSModel.load(str(PROJECT_ROOT / cfg["paths"]["als_model"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện Spark ALS trên MovieLens 32M")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    parser.add_argument("--tune", action="store_true", help="Grid search rank x regParam trước khi huấn luyện")
    args = parser.parse_args()

    _, train_data, _ = load_and_split_ratings(args.env)
    overrides: Dict[str, Any] = {}
    if args.tune:
        best = tune_als_hyperparameters(train_data, args.env)
        overrides = {"rank": best["rank"], "reg_param": best["reg_param"]}

    model, _ = train_als_model(train_data, args.env, **overrides)
    save_als_model(model, args.env)


if __name__ == "__main__":
    main()
