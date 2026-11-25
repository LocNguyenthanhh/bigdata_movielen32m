"""Đánh giá chất lượng thuật toán ALS: RMSE, Precision@K, NDCG@K & Cold-start analysis.

Quy ước đánh giá:
    - RMSE: trên các cặp (user, movie) của tập test mà cả user và movie đều có trong train
      (coldStartStrategy = "drop").
    - Precision@K / NDCG@K: với mỗi user trong test, tập "phim liên quan" là các phim user
      chấm >= `evaluation.relevance_threshold` trong test. Danh sách gợi ý là Top-K của ALS
      sau khi loại các phim user đã xem trong train (left_anti).
    - Baseline Bayesian Popularity: cùng giao thức trên nhưng mọi user nhận cùng một bảng
      phim phổ biến → cho thấy mức cải thiện nhờ cá nhân hóa của ALS.
    - Cold-start: thống kê tỷ lệ dòng test có user / movie chưa xuất hiện trong train, và RMSE
      khi dùng gợi ý dự phòng (điểm Bayes của phim, hoặc trung bình toàn cục) cho các dòng đó.

Phụ trách: Nguyễn Văn Tâm (Spark MLlib Modeling)
"""

import json
from typing import Any, Dict, List, Optional

from pyspark.ml.evaluation import RankingEvaluator, RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.config import PROJECT_ROOT, load_config
from src.ml.cold_start import compute_bayesian_popularity


def _rmse(df: DataFrame, label_col: str, prediction_col: str) -> Optional[float]:
    """RMSE trên DataFrame; trả về None nếu rỗng."""
    if df.limit(1).count() == 0:
        return None
    evaluator = RegressionEvaluator(metricName="rmse", labelCol=label_col, predictionCol=prediction_col)
    return round(float(evaluator.evaluate(df)), 4)


def _topk_lists(candidates: DataFrame, train_df: DataFrame, k: int) -> DataFrame:
    """Từ bảng ứng viên (user_id, movie_id, score): loại phim đã xem trong train, giữ Top-K.

    Returns:
        DataFrame (user_id, recommendations: array<double>) theo thứ tự score giảm dần.
    """
    watched = train_df.select("user_id", "movie_id").distinct()
    window_spec = Window.partitionBy("user_id").orderBy(F.col("score").desc(), F.col("movie_id"))
    ranked = (
        candidates.join(watched, on=["user_id", "movie_id"], how="left_anti")
        .withColumn("rank", F.row_number().over(window_spec))
        .filter(F.col("rank") <= k)
    )
    return ranked.groupBy("user_id").agg(
        F.expr(
            "transform(sort_array(collect_list(struct(rank, movie_id))), x -> cast(x.movie_id as double))"
        ).alias("recommendations")
    )


def _ranking_metrics(pred_lists: DataFrame, relevant_df: DataFrame, k_values: List[int]) -> Dict[str, float]:
    """Tính Precision@K và NDCG@K bằng RankingEvaluator của Spark MLlib."""
    joined = relevant_df.join(pred_lists, on="user_id", how="inner").cache()
    metrics: Dict[str, float] = {"ranking_eval_users": joined.count()}
    if metrics["ranking_eval_users"] == 0:
        joined.unpersist()
        return metrics

    for k in k_values:
        for metric_name, key in (("precisionAtK", f"precision@{k}"), ("ndcgAtK", f"ndcg@{k}")):
            evaluator = RankingEvaluator(
                metricName=metric_name,
                k=k,
                predictionCol="recommendations",
                labelCol="relevant",
            )
            metrics[key] = round(float(evaluator.evaluate(joined)), 4)
    joined.unpersist()
    return metrics


def analyze_cold_start(
    train_df: DataFrame,
    test_df: DataFrame,
    popularity_df: DataFrame,
) -> Dict[str, Any]:
    """Phân tích Cold-start: tỷ lệ user/movie mới trong test & RMSE của gợi ý dự phòng."""
    train_users = train_df.select("user_id").distinct().withColumn("known_user", F.lit(True))
    train_items = train_df.select("movie_id").distinct().withColumn("known_item", F.lit(True))
    global_mean = float(train_df.agg(F.avg("rating")).first()[0])

    flagged = (
        test_df.join(train_users, on="user_id", how="left")
        .join(train_items, on="movie_id", how="left")
        .withColumn("cold_user", F.col("known_user").isNull())
        .withColumn("cold_item", F.col("known_item").isNull())
        .cache()
    )

    stats = flagged.agg(
        F.count("*").alias("test_rows"),
        F.sum(F.col("cold_user").cast("int")).alias("cold_user_rows"),
        F.sum(F.col("cold_item").cast("int")).alias("cold_item_rows"),
        F.sum((F.col("cold_user") | F.col("cold_item")).cast("int")).alias("cold_rows"),
        F.countDistinct(F.when(F.col("cold_user"), F.col("user_id"))).alias("cold_users"),
        F.countDistinct("user_id").alias("test_users"),
    ).first().asDict()

    test_rows = max(int(stats["test_rows"] or 0), 1)
    cold_rows = flagged.filter(F.col("cold_user") | F.col("cold_item"))

    # Gợi ý dự phòng: điểm Bayes của phim (nếu phim đã biết) hoặc trung bình toàn cục
    fallback = (
        cold_rows.join(popularity_df.select("movie_id", "bayes_score"), on="movie_id", how="left")
        .withColumn("prediction", F.coalesce(F.col("bayes_score"), F.lit(global_mean)))
        .withColumn("global_mean_pred", F.lit(global_mean))
    )

    result = {
        "test_rows": int(stats["test_rows"] or 0),
        "test_users": int(stats["test_users"] or 0),
        "cold_users": int(stats["cold_users"] or 0),
        "cold_user_rows": int(stats["cold_user_rows"] or 0),
        "cold_item_rows": int(stats["cold_item_rows"] or 0),
        "cold_rows": int(stats["cold_rows"] or 0),
        "cold_rows_ratio": round(int(stats["cold_rows"] or 0) / test_rows, 6),
        "global_mean_rating": round(global_mean, 4),
        "rmse_cold_rows_bayesian_fallback": _rmse(fallback, "rating", "prediction"),
        "rmse_cold_rows_global_mean": _rmse(fallback, "rating", "global_mean_pred"),
    }
    flagged.unpersist()
    print(
        f"[COLD-START] {result['cold_rows']}/{result['test_rows']} dòng test "
        f"({result['cold_rows_ratio']:.2%}) có user/movie mới | {result['cold_users']} user mới | "
        f"RMSE fallback Bayes = {result['rmse_cold_rows_bayesian_fallback']}"
    )
    return result


def evaluate_als_model(
    model: ALSModel,
    test_df: DataFrame,
    train_df: Optional[DataFrame] = None,
    train_time_sec: float = 0.0,
    env: str = "dev",
    popularity_df: Optional[DataFrame] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Đánh giá đầy đủ mô hình ALS và lưu kết quả vào `paths.metrics_json`.

    Args:
        model: Mô hình ALS đã huấn luyện.
        test_df: Tập kiểm thử (user_id, movie_id, rating).
        train_df: Tập huấn luyện. Nếu None chỉ tính RMSE.
        train_time_sec: Thời gian huấn luyện (ghi vào metrics).
        env: Môi trường cấu hình.
        popularity_df: Bảng Bayesian Popularity tính trên train (nếu None sẽ tự tính).
        extra_info: Thông tin bổ sung ghi vào metrics (ví dụ kết quả tuning).
    """
    cfg = load_config(env)
    eval_cfg = cfg["evaluation"]
    k_values = [int(k) for k in eval_cfg.get("k_values", [10, 20])]
    max_k = max(k_values)
    threshold = float(eval_cfg.get("relevance_threshold", 4.0))
    n_candidates = max_k + int(eval_cfg.get("ranking_candidates", 100))

    metrics: Dict[str, Any] = {
        "environment": env,
        "sample_fraction": cfg.get("data", {}).get("sample_fraction", 1.0),
        "rank": model.rank,
        "max_iter": int(cfg["als"]["max_iter"]),
        "reg_param": float(cfg["als"]["reg_param"]),
        "train_time_sec": round(train_time_sec, 2),
    }
    if extra_info:
        metrics.update(extra_info)

    # 1) RMSE
    predictions = model.transform(test_df)
    metrics["rmse"] = _rmse(predictions, cfg["als"]["rating_col"], "prediction")
    print(f"[EVAL] RMSE = {metrics['rmse']}")

    if train_df is not None:
        if popularity_df is None:
            popularity_df = compute_bayesian_popularity(train_df, env=env, save=False)

        # 2) Precision@K / NDCG@K của ALS
        relevant_df = (
            test_df.filter(F.col("rating") >= threshold)
            .groupBy("user_id")
            .agg(F.collect_set(F.col("movie_id").cast("double")).alias("relevant"))
            .cache()
        )
        eval_users = relevant_df.select("user_id")

        als_candidates = model.recommendForUserSubset(eval_users, n_candidates).select(
            "user_id", F.explode("recommendations").alias("rec")
        ).select("user_id", F.col("rec.movie_id").alias("movie_id"), F.col("rec.rating").alias("score"))
        als_ranking = _ranking_metrics(_topk_lists(als_candidates, train_df, max_k), relevant_df, k_values)
        metrics["als_ranking"] = als_ranking
        print(f"[EVAL] ALS ranking: {als_ranking}")

        # 3) Baseline Bayesian Popularity (không cá nhân hóa)
        top_popular = popularity_df.orderBy(F.col("bayes_score").desc()).limit(n_candidates).select(
            "movie_id", F.col("bayes_score").alias("score")
        )
        pop_candidates = eval_users.crossJoin(F.broadcast(top_popular))
        pop_ranking = _ranking_metrics(_topk_lists(pop_candidates, train_df, max_k), relevant_df, k_values)
        metrics["popularity_baseline_ranking"] = pop_ranking
        print(f"[EVAL] Popularity baseline ranking: {pop_ranking}")
        relevant_df.unpersist()

        # 4) Cold-start analysis
        metrics["cold_start"] = analyze_cold_start(train_df, test_df, popularity_df)

    metrics_path = PROJECT_ROOT / cfg["paths"]["metrics_json"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[EVAL] Đã lưu metrics tại {metrics_path}")
    return metrics


if __name__ == "__main__":
    import argparse

    from src.ml.train_als import load_and_split_ratings, train_als_model

    parser = argparse.ArgumentParser(description="Đánh giá Spark ALS trên MovieLens 32M")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    args = parser.parse_args()

    _, train_data, test_data = load_and_split_ratings(args.env)
    als_model, elapsed = train_als_model(train_data, args.env)
    evaluate_als_model(als_model, test_data, train_df=train_data, train_time_sec=elapsed, env=args.env)
