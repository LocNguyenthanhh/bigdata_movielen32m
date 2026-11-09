"""Đo thời gian thực thi khi thay đổi số nhân xử lý (1, 2, 4 cores) -> Kiểm chứng Định luật Amdahl.

Phụ trách: Nguyễn Thanh Lộc (HPC Benchmarking & Lead Report)
"""

import csv
import time
from typing import List

from src.common.config import PROJECT_ROOT, load_config
from src.common.spark_session import get_spark_session
from src.ml.train_als import train_als_model


def run_speedup_benchmark(cores_list: List[int] = None, env: str = "dev") -> None:
    """Chạy benchmark theo số Cores và ghi kết quả Speedup / Efficiency vào CSV."""
    cores_to_test = cores_list or [1, 2, 4]
    cfg = load_config(env)
    ratings_path = str(PROJECT_ROOT / cfg["paths"]["ratings_parquet"])
    csv_path = PROJECT_ROOT / cfg["paths"]["benchmark_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    sample_fraction = float(cfg.get("data", {}).get("sample_fraction", 0.05))
    seed = int(cfg["project"]["seed"])

    results = []
    t1 = None

    for cores in cores_to_test:
        spark = get_spark_session(cfg, master_override=f"local[{cores}]", app_name_override=f"Benchmark-Cores-{cores}")
        df = spark.read.parquet(ratings_path)
        if sample_fraction < 1.0:
            df = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)

        start = time.perf_counter()
        _, train_time = train_als_model(df, env=env)
        total_time = time.perf_counter() - start
        spark.stop()

        if t1 is None:
            t1 = total_time
        speedup = t1 / total_time if total_time > 0 else 1.0
        efficiency = speedup / cores

        results.append({
            "benchmark_type": "speedup",
            "cores": cores,
            "sample_fraction": sample_fraction,
            "als_train_sec": round(train_time, 2),
            "total_sec": round(total_time, 2),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 3),
        })
        print(f"[BENCHMARK SPEEDUP] Cores={cores} | Time={total_time:.2f}s | Speedup={speedup:.2f}x | Eff={efficiency:.2%}")

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    run_speedup_benchmark()
