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
from src.common.config import PROJECT_ROOT, load_config
from src.ml.clustering import run_clustering
from src.ml.cold_start import compute_bayesian_popularity
from src.ml.evaluate_model import evaluate_als_model
from src.ml.generate_topn import generate_topn_recommendations
from src.ml.train_als import load_and_split_ratings, save_als_model, train_als_model, tune_als_hyperparameters


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
    parser.add_argument(
        "--with-tuning",
        action="store_true",
        help="Grid search rank x regParam của ALS trước khi huấn luyện mô hình cuối.",
    )
    parser.add_argument(
        "--with-clustering",
        action="store_true",
        help="(Bonus) Phân cụm phim K-Means trên ALS itemFactors.",
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
    train_df = train_df.cache()

    overrides, extra_info = {}, {}
    if args.with_tuning:
        best = tune_als_hyperparameters(train_df, env=env)
        overrides = {"rank": best["rank"], "reg_param": best["reg_param"]}
        extra_info = {"reg_param": best["reg_param"], "tuning_best": best}

    model, train_time = train_als_model(train_df, env=env, **overrides)
    save_als_model(model, env=env)

    popularity_df = compute_bayesian_popularity(train_df, env=env)
    evaluate_als_model(
        model, test_df, train_df=train_df, train_time_sec=train_time, env=env,
        popularity_df=popularity_df, extra_info=extra_info,
    )
    generate_topn_recommendations(model, ratings_df, env=env)

    if args.with_clustering:
        movies_df = ratings_df.sparkSession.read.parquet(str(PROJECT_ROOT / load_config(env)["paths"]["movies_parquet"]))
        run_clustering(model, train_df, movies_df, env=env)

    if args.with_benchmark:
        run_speedup_benchmark(env=env)
        run_scaling_benchmark(env=env)
        plot_benchmark_results(env=env)

    print("=== HOÀN TẤT PIPELINE MOVIELENS 32M ===")


if __name__ == "__main__":
    main()
