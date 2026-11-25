"""(Bonus) Phân cụm phim bằng K-Means trên vector tiềm ẩn `itemFactors` của Spark ALS.

Mỗi phim được ALS biểu diễn bằng một vector `rank` chiều. Các phim có vector gần nhau
được cùng nhóm người dùng yêu thích → K-Means trên không gian này gom được các nhóm phim
có "phong cách" tương đồng mà không cần dùng thể loại (genres). Thể loại chỉ dùng để
diễn giải từng cụm sau khi phân cụm.

Phụ trách: Nguyễn Văn Tâm (Spark MLlib Modeling)

Chạy độc lập (cần mô hình đã lưu bởi `python -m src.ml.train_als`):
    python -m src.ml.clustering --env dev
"""

from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import array_to_vector
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.config import PROJECT_ROOT, load_config


def extract_item_factors(
    model: ALSModel,
    ratings_df: Optional[DataFrame] = None,
    min_item_ratings: int = 0,
) -> DataFrame:
    """Lấy `itemFactors` thành DataFrame (movie_id, features: Vector).

    Nếu truyền `ratings_df`, chỉ giữ phim có >= `min_item_ratings` lượt đánh giá
    (vector của phim quá ít rating rất nhiễu, làm cụm kém ý nghĩa).
    """
    factors = model.itemFactors.select(
        F.col("id").alias("movie_id"),
        array_to_vector(F.col("features")).alias("features"),
    )
    if ratings_df is not None and min_item_ratings > 0:
        popular_ids = (
            ratings_df.groupBy("movie_id").agg(F.count("*").alias("num_ratings"))
            .filter(F.col("num_ratings") >= min_item_ratings)
            .select("movie_id")
        )
        factors = factors.join(popular_ids, on="movie_id", how="inner")
    return factors


def scan_k(factors_df: DataFrame, k_values: List[int], seed: int = 42) -> List[Dict[str, Any]]:
    """Chạy K-Means với nhiều giá trị k; trả về Silhouette và WSSSE (phục vụ Elbow method)."""
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                    metricName="silhouette", distanceMeasure="squaredEuclidean")
    factors_df = factors_df.cache()
    results = []
    for k in k_values:
        model = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster").fit(factors_df)
        silhouette = float(evaluator.evaluate(model.transform(factors_df)))
        wssse = float(model.summary.trainingCost)
        results.append({"k": k, "silhouette": round(silhouette, 4), "wssse": round(wssse, 2)})
        print(f"[KMEANS] k={k:<3} silhouette={silhouette:.4f} WSSSE={wssse:.2f}")
    factors_df.unpersist()
    return results


def fit_kmeans(factors_df: DataFrame, k: int, seed: int = 42) -> Tuple[KMeansModel, DataFrame]:
    """Huấn luyện K-Means và gán cụm cho từng phim -> (model, DataFrame(movie_id, features, cluster))."""
    model = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster").fit(factors_df)
    return model, model.transform(factors_df)


def profile_clusters(
    clustered_df: DataFrame,
    movies_df: DataFrame,
    ratings_df: DataFrame,
    top_genres: int = 3,
    top_titles: int = 5,
) -> DataFrame:
    """Diễn giải từng cụm: kích thước, thể loại nổi bật (tỷ lệ), các phim nhiều lượt xem nhất.

    Returns:
        DataFrame (cluster, size, top_genres, sample_titles) sắp theo cluster.
    """
    movie_info = movies_df.select("movie_id", "title", "genres")
    counts = ratings_df.groupBy("movie_id").agg(F.count("*").alias("num_ratings"))
    enriched = clustered_df.select("movie_id", "cluster").join(movie_info, "movie_id", "left") \
        .join(counts, "movie_id", "left").cache()

    sizes = enriched.groupBy("cluster").agg(F.count("*").alias("size"))

    genre_share = (
        enriched.select("cluster", F.explode_outer("genres").alias("genre"))
        .filter(F.col("genre").isNotNull() & (F.col("genre") != "(no genres listed)"))
        .groupBy("cluster", "genre").agg(F.count("*").alias("n"))
        .join(sizes, "cluster")
        .withColumn("share", F.round(F.col("n") / F.col("size"), 2))
        .withColumn("r", F.row_number().over(Window.partitionBy("cluster").orderBy(F.col("n").desc())))
        .filter(F.col("r") <= top_genres)
        .groupBy("cluster")
        .agg(F.expr("concat_ws(', ', transform(sort_array(collect_list(struct(r, genre, share))), "
                    "x -> concat(x.genre, ' (', cast(x.share as string), ')')))").alias("top_genres"))
    )

    samples = (
        enriched.withColumn(
            "r", F.row_number().over(Window.partitionBy("cluster").orderBy(F.col("num_ratings").desc_nulls_last()))
        )
        .filter(F.col("r") <= top_titles)
        .groupBy("cluster")
        .agg(F.expr("concat_ws(' | ', transform(sort_array(collect_list(struct(r, title))), x -> x.title))")
             .alias("sample_titles"))
    )

    profile = sizes.join(genre_share, "cluster", "left").join(samples, "cluster", "left").orderBy("cluster")
    enriched.unpersist()
    return profile


def run_clustering(model: ALSModel, ratings_df: DataFrame, movies_df: DataFrame, env: str = "dev") -> DataFrame:
    """Pipeline bonus: itemFactors -> K-Means -> lưu cụm & in mô tả từng cụm."""
    cfg = load_config(env)
    cl_cfg = cfg.get("clustering", {})
    seed = int(cfg["project"]["seed"])

    factors = extract_item_factors(model, ratings_df, int(cl_cfg.get("min_item_ratings", 20)))
    _, clustered = fit_kmeans(factors, int(cl_cfg.get("k", 10)), seed)

    out_path = str(PROJECT_ROOT / cfg["paths"]["clusters_parquet"])
    clustered.select("movie_id", "cluster").write.mode("overwrite").parquet(out_path)
    print(f"[KMEANS] Đã lưu nhãn cụm phim tại {out_path}")

    profile = profile_clusters(clustered, movies_df, ratings_df)
    profile.show(truncate=False)
    return profile


if __name__ == "__main__":
    import argparse

    from src.common.spark_session import get_spark_session
    from src.ml.train_als import load_als_model

    parser = argparse.ArgumentParser(description="K-Means trên ALS itemFactors")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    args = parser.parse_args()

    config = load_config(args.env)
    spark = get_spark_session(config)
    ratings = spark.read.parquet(str(PROJECT_ROOT / config["paths"]["ratings_parquet"]))
    movies = spark.read.parquet(str(PROJECT_ROOT / config["paths"]["movies_parquet"]))
    run_clustering(load_als_model(args.env), ratings, movies, args.env)
