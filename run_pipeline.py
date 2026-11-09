"""Script điều phối chạy toàn bộ Pipeline: Download -> ETL -> Train ALS -> Evaluate -> TopN -> Benchmark.

Cách dùng:
    python run_pipeline.py --env dev
    python run_pipeline.py --env prod --with-benchmark
"""

import argparse

from src.benchmarks.benchmark_scaling import run_scaling_benchmark
from src.benchmarks.benchmark_speedup import run_speedup_benchmark
from src.benchmarks.plot_benchmarks import plot_benchmark_results
from src.etl.csv_to_parquet import run_etl
from src.etl.download_data import download_and_extract
from src.ml.evaluate_model import evaluate_als_model
from src.ml.generate_topn import generate_topn_recommendations
from src.ml.train_als import load_and_split_ratings, train_als_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens 32M Recommendation & HPC Benchmark Pipeline")
    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        choices=["dev", "prod"],
        help="Môi trường cấu hình: 'dev' (5%% dữ liệu) hoặc 'prod' (100%% dữ liệu 32M).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Bỏ qua bước tải dữ liệu nếu đã có sẵn trong data/raw/.",
    )
    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help="Bỏ qua bước chuyển đổi CSV sang Parquet nếu đã có sẵn trong data/lake/.",
    )
    parser.add_argument(
        "--with-benchmark",
        action="store_true",
        help="Chạy thêm bài đo Speedup (Định luật Amdahl), Data Scaling và vẽ biểu đồ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = args.env
    print(f"=== BẮT ĐẦU PIPELINE MOVIELENS 32M (MÔI TRƯỜNG: {env.upper()}) ===")

    if not args.skip_download:
        download_and_extract(env=env)

    if not args.skip_etl:
        run_etl(env=env)

    ratings_df, train_df, test_df = load_and_split_ratings(env=env)
    model, train_time = train_als_model(train_df, env=env)
    evaluate_als_model(model, test_df, train_time_sec=train_time, env=env)
    generate_topn_recommendations(model, ratings_df, env=env)

    if args.with_benchmark:
        run_speedup_benchmark(env=env)
        run_scaling_benchmark(env=env)
        plot_benchmark_results(env=env)

    print("=== HOÀN TẤT PIPELINE MOVIELENS 32M ===")


if __name__ == "__main__":
    main()
