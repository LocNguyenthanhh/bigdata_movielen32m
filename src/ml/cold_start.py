"""Xử lý Cold-start bằng Bayesian Popularity (gợi ý dự phòng cho người dùng mới).

Người dùng mới chưa có lịch sử đánh giá nên ALS không có vector tiềm ẩn cho họ.
Khi đó hệ thống gợi ý các phim "phổ biến & được đánh giá cao" theo điểm Bayes
(Weighted Rating, công thức kiểu IMDb):

    WR = (v / (v + m)) * R + (m / (v + m)) * C

    - v: số lượt đánh giá của phim
    - R: điểm trung bình của phim
    - C: điểm trung bình toàn cục trên mọi rating
    - m: số lượt đánh giá tối thiểu (prior), lấy theo phân vị `min_votes_quantile`
         của phân phối số lượt đánh giá mỗi phim

Phim ít lượt đánh giá bị "kéo" về C, tránh việc phim có 1-2 rating 5.0 lên đầu bảng.

Phụ trách: Nguyễn Văn Tâm (Spark MLlib Modeling)
"""

from typing import Iterable, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.config import PROJECT_ROOT, load_config


def votes_quantile(movie_stats: DataFrame, quantile: float) -> float:
    """Phân vị `quantile` của số lượt đánh giá mỗi phim (cột `num_ratings`), tối thiểu 1."""
    return max(float(movie_stats.approxQuantile("num_ratings", [quantile], 0.01)[0]), 1.0)


def compute_bayesian_popularity(ratings_df: DataFrame, env: str = "dev", save: bool = True) -> DataFrame:
    """Tính bảng xếp hạng phim theo Bayesian Popularity và (tuỳ chọn) lưu ra Parquet.

    Returns:
        DataFrame gồm `movie_id`, `num_ratings`, `avg_rating`, `bayes_score`, `rank`.
    """
    cfg = load_config(env)
    cs_cfg = cfg.get("cold_start", {})
    quantile = float(cs_cfg.get("min_votes_quantile", 0.90))

    movie_stats = ratings_df.groupBy("movie_id").agg(
        F.count("rating").alias("num_ratings"),
        F.avg("rating").alias("avg_rating"),
    ).cache()

    global_mean = ratings_df.agg(F.avg("rating")).first()[0]
    min_votes = votes_quantile(movie_stats, quantile)

    v = F.col("num_ratings").cast("double")
    m = F.lit(min_votes)
    popularity_df = movie_stats.withColumn(
        "bayes_score",
        (v / (v + m)) * F.col("avg_rating") + (m / (v + m)) * F.lit(global_mean),
    )

    window_spec = Window.orderBy(F.col("bayes_score").desc(), F.col("num_ratings").desc())
    popularity_df = popularity_df.withColumn("rank", F.row_number().over(window_spec))

    print(
        f"[COLD-START] Bayesian Popularity: C (global mean) = {global_mean:.4f}, "
        f"m (q={quantile:.2f}) = {min_votes:.0f} lượt đánh giá"
    )

    if save:
        output_path = str(PROJECT_ROOT / cfg["paths"]["popular_parquet"])
        popularity_df.write.mode("overwrite").parquet(output_path)
        print(f"[COLD-START] Đã lưu bảng phim phổ biến tại {output_path}")

    movie_stats.unpersist()
    return popularity_df


def recommend_for_new_user(
    popularity_df: DataFrame,
    top_n: int = 20,
    watched_movie_ids: Optional[Iterable[int]] = None,
) -> DataFrame:
    """Gợi ý Top-N phim cho người dùng mới (Cold-start), bỏ qua các phim đã xem nếu có."""
    candidates = popularity_df
    watched = list(watched_movie_ids or [])
    if watched:
        candidates = candidates.filter(~F.col("movie_id").isin(watched))
    return candidates.orderBy(F.col("bayes_score").desc()).limit(top_n)
