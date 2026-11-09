"""PySpark ETL: Đọc CSV -> Chuẩn hóa Schema -> Ghi Parquet nén Snappy.

Phụ trách: Nguyễn Đinh Nhật Quang (Data Engineering)
"""

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, LongType, StringType, StructField, StructType

from src.common.config import PROJECT_ROOT, load_config
from src.common.spark_session import get_spark_session


RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), nullable=False),
    StructField("movieId", IntegerType(), nullable=False),
    StructField("rating", FloatType(), nullable=False),
    StructField("timestamp", LongType(), nullable=False),
])

MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), nullable=False),
    StructField("title", StringType(), nullable=True),
    StructField("genres", StringType(), nullable=True),
])


def run_etl(env: str = "dev") -> None:
    """Chuyển đổi ratings.csv và movies.csv sang định dạng Apache Parquet."""
    cfg = load_config(env)
    spark = get_spark_session(cfg)

    raw_folder = PROJECT_ROOT / cfg["paths"]["raw_dir"] / cfg["dataset"]["extracted_folder"]
    ratings_out = str(PROJECT_ROOT / cfg["paths"]["ratings_parquet"])
    movies_out = str(PROJECT_ROOT / cfg["paths"]["movies_parquet"])
    compression = cfg["etl"].get("compression", "snappy")

    print(f"[ETL] Đọc dữ liệu từ {raw_folder} ...")
    ratings_df = (
        spark.read.csv(str(raw_folder / "ratings.csv"), header=True, schema=RATINGS_SCHEMA)
        .withColumnRenamed("userId", "user_id")
        .withColumnRenamed("movieId", "movie_id")
        .withColumn("year", F.year(F.from_unixtime(F.col("timestamp"))))
    )

    movies_df = (
        spark.read.csv(str(raw_folder / "movies.csv"), header=True, schema=MOVIES_SCHEMA)
        .withColumnRenamed("movieId", "movie_id")
        .withColumn("release_year", F.regexp_extract(F.col("title"), r"\((\d{4})\)", 1).cast(IntegerType()))
        .withColumn("genres", F.split(F.col("genres"), r"\|"))
    )

    print(f"[ETL] Ghi bảng movies ra Parquet tại {movies_out} ...")
    movies_df.write.mode("overwrite").option("compression", compression).parquet(movies_out)

    print(f"[ETL] Ghi bảng ratings (partition by year) ra Parquet tại {ratings_out} ...")
    (
        ratings_df.write.mode("overwrite")
        .partitionBy("year")
        .option("compression", compression)
        .parquet(ratings_out)
    )
    print("[ETL] Hoàn tất chuyển đổi CSV sang Parquet Data Lake.")


if __name__ == "__main__":
    run_etl()
