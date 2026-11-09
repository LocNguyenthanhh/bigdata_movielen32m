"""Đánh giá chất lượng thuật toán ALS: RMSE, Precision@K, NDCG@K & Cold-start analysis.

Phụ trách: Nguyễn Văn Tâm (Spark MLlib Modeling)
"""

import json
from typing import Dict
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame

from src.common.config import PROJECT_ROOT, load_config


def evaluate_als_model(
    model: ALSModel,
    test_df: DataFrame,
    train_time_sec: float = 0.0,
    env: str = "dev",
) -> Dict[str, float]:
    """Đánh giá RMSE trên tập kiểm thử và lưu kết quả vào data/output/metrics.json."""
    cfg = load_config(env)
    predictions = model.transform(test_df)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol=cfg["als"]["rating_col"],
        predictionCol="prediction",
    )
    rmse = float(evaluator.evaluate(predictions))

    metrics = {
        "environment": env,
        "sample_fraction": cfg.get("data", {}).get("sample_fraction", 1.0),
        "rank": cfg["als"]["rank"],
        "max_iter": cfg["als"]["max_iter"],
        "reg_param": cfg["als"]["reg_param"],
        "rmse": round(rmse, 4),
        "train_time_sec": round(train_time_sec, 2),
    }

    metrics_path = PROJECT_ROOT / cfg["paths"]["metrics_json"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[EVAL] RMSE = {rmse:.4f} | Đã lưu metrics tại {metrics_path}")
    return metrics
