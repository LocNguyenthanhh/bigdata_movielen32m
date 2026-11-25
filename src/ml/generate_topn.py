"""Sinh Top-N bộ phim gợi ý cho từng người dùng (khử phim đã xem bằng left_anti).

Phụ trách: Nguyễn Văn Tâm (Spark MLlib Modeling)
"""

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.config import PROJECT_ROOT, load_config
from src.ml.cold_start import votes_quantile


def generate_topn_recommendations(
    model: ALSModel,
    ratings_df: DataFrame,
    env: str = "dev",
) -> DataFrame:
    """Sinh bảng gợi ý Top-N phim chưa xem cho mỗi user và lưu ra Parquet.

    Output: (user_id, movie_id, score, rank) theo Data Contract.
    """
    cfg = load_config(env)
    top_n = int(cfg["als"].get("top_n", 20))
    min_q = float(cfg["als"].get("candidate_min_ratings_quantile", 0.0))
    extra = cfg["evaluation"].get("filtered_candidates" if min_q > 0 else "ranking_candidates", 100)
    n_candidates = top_n + int(extra)
    output_path = str(PROJECT_ROOT / cfg["paths"]["topn_parquet"])

    # Lấy dư ứng viên vì một phần sẽ bị loại do user đã xem (và do lọc phim ít lượt đánh giá)
    raw_recs = model.recommendForAllUsers(n_candidates)

    exploded_recs = raw_recs.select(
        F.col("user_id"),
        F.explode("recommendations").alias("rec"),
    ).select(
        F.col("user_id"),
        F.col("rec.movie_id").alias("movie_id"),
        F.col("rec.rating").alias("score"),
    )

    # Chỉ gợi ý phim có đủ lượt đánh giá (ALS hay đẩy phim rất ít dữ liệu lên đầu)
    if min_q > 0:
        movie_stats = ratings_df.groupBy("movie_id").agg(F.count("*").alias("num_ratings"))
        min_votes = votes_quantile(movie_stats, min_q)
        eligible = movie_stats.filter(F.col("num_ratings") >= min_votes).select("movie_id")
        exploded_recs = exploded_recs.join(F.broadcast(eligible), "movie_id")
        print(f"[TOP-N] Chỉ gợi ý phim có >= {min_votes:.0f} lượt đánh giá")

    # Khử các bộ phim người dùng đã xem (left_anti join)
    watched_df = ratings_df.select("user_id", "movie_id").distinct()
    unwatched_recs = exploded_recs.join(watched_df, on=["user_id", "movie_id"], how="left_anti")

    window_spec = Window.partitionBy("user_id").orderBy(F.col("score").desc(), F.col("movie_id"))
    topn_df = (
        unwatched_recs.withColumn("rank", F.row_number().over(window_spec))
        .filter(F.col("rank") <= top_n)
        .select("user_id", "movie_id", F.col("score").cast("float").alias("score"), "rank")
    )

    topn_df.write.mode("overwrite").parquet(output_path)
    print(f"[TOP-N] Đã lưu Top-{top_n} gợi ý tại {output_path}")
    return topn_df


if __name__ == "__main__":
    import argparse

    from src.ml.train_als import load_als_model, load_and_split_ratings

    parser = argparse.ArgumentParser(description="Sinh Top-N gợi ý từ mô hình ALS đã lưu")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    args = parser.parse_args()

    all_ratings, _, _ = load_and_split_ratings(args.env)
    generate_topn_recommendations(load_als_model(args.env), all_ratings, args.env)
