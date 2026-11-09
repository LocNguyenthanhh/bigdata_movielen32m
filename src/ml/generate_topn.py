"""Sinh Top-N bộ phim gợi ý cho từng người dùng (khử phim đã xem bằng left_anti).

Phụ trách: Nguyễn Văn Tâm (Spark MLlib Modeling)
"""

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.config import PROJECT_ROOT, load_config


def generate_topn_recommendations(
    model: ALSModel,
    ratings_df: DataFrame,
    env: str = "dev",
) -> DataFrame:
    """Sinh bảng gợi ý Top-N phim chưa xem cho mỗi user và lưu ra Parquet."""
    cfg = load_config(env)
    top_n = int(cfg["als"].get("top_n", 20))
    output_path = str(PROJECT_ROOT / cfg["paths"]["topn_parquet"])

    # Lấy nhiều hơn top_n một chút trước khi lọc các phim đã xem
    raw_recs = model.recommendForAllUsers(top_n * 2)

    exploded_recs = raw_recs.select(
        F.col("user_id"),
        F.explode("recommendations").alias("rec"),
    ).select(
        F.col("user_id"),
        F.col("rec.movie_id").alias("movie_id"),
        F.col("rec.rating").alias("score"),
    )

    # Khử các bộ phim người dùng đã xem (left_anti join)
    watched_df = ratings_df.select("user_id", "movie_id").distinct()
    unwatched_recs = exploded_recs.join(watched_df, on=["user_id", "movie_id"], how="left_anti")

    window_spec = Window.partitionBy("user_id").orderBy(F.col("score").desc())
    topn_df = (
        unwatched_recs.withColumn("rank", F.row_number().over(window_spec))
        .filter(F.col("rank") <= top_n)
    )

    topn_df.write.mode("overwrite").parquet(output_path)
    print(f"[TOP-N] Đã lưu Top-{top_n} gợi ý tại {output_path}")
    return topn_df
