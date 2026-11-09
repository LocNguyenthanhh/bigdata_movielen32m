"""Đo thời gian thực thi khi mở rộng kích thước dữ liệu (5%, 20%, 50%, 100%).

Phụ trách: Nguyễn Thanh Lộc (HPC Benchmarking & Lead Report)
"""

import csv
import time
from typing import List

from src.common.config import PROJECT_ROOT, load_config
from src.common.spark_session import get_spark_session
from src.ml.train_als import train_als_model


def run_scaling_benchmark(fractions: List[float] = None, env: str = "prod") -> None:
    """Đánh giá khả năng mở rộng theo quy mô dữ liệu (Data Scalability)."""
    fractions_to_test = fractions or [0.05, 0.20, 0.50, 1.00]
    cfg = load_config(env)
    spark = get_spark_session(cfg)

    ratings_path = str(PROJECT_ROOT / cfg["paths"]["ratings_parquet"])
    csv_path = PROJECT_ROOT / cfg["paths"]["benchmark_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["project"]["seed"])

    full_df = spark.read.parquet(ratings_path)
    results = []

    for frac in fractions_to_test:
        sub_df = full_df.sample(False, frac, seed=seed) if frac < 1.0 else full_df
        start = time.perf_counter()
        _, train_time = train_als_model(sub_df, env=env)
        total_time = time.perf_counter() - start

        results.append({
            "benchmark_type": "scaling",
            "cores": cfg["spark"]["master"],
            "sample_fraction": frac,
            "als_train_sec": round(train_time, 2),
            "total_sec": round(total_time, 2),
            "speedup": 1.0,
            "efficiency": 1.0,
        })
        print(f"[BENCHMARK SCALING] Fraction={frac:.0%} | Time={total_time:.2f}s")

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    run_scaling_benchmark()
