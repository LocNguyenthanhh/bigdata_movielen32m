"""Tự động xuất biểu đồ Speedup (Định luật Amdahl) & Data Scalability cho Báo cáo.

Phụ trách: Nguyễn Thanh Lộc (HPC Benchmarking & Lead Report)
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import PROJECT_ROOT, load_config


def plot_benchmark_results(env: str = "dev") -> None:
    """Đọc benchmark_results.csv và xuất biểu đồ PNG vào reports/figures/."""
    cfg = load_config(env)
    csv_path = PROJECT_ROOT / cfg["paths"]["benchmark_csv"]
    figures_dir = PROJECT_ROOT / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[PLOT] Chưa tìm thấy file {csv_path}. Hãy chạy benchmark trước.")
        return

    df = pd.read_csv(csv_path)

    # 1. Biểu đồ Speedup (Định luật Amdahl)
    speedup_df = df[df["benchmark_type"] == "speedup"].copy()
    if not speedup_df.empty:
        speedup_df["cores"] = speedup_df["cores"].astype(int)
        speedup_df = speedup_df.sort_values("cores")

        plt.figure(figsize=(7, 5))
        plt.plot(speedup_df["cores"], speedup_df["speedup"], marker="o", linewidth=2, label="Thực nghiệm Spark ALS")
        plt.plot(speedup_df["cores"], speedup_df["cores"], linestyle="--", color="gray", label="Tuyến tính lý tưởng ($S_p = p$)")
        plt.title("Đánh giá Tăng tốc (Speedup) theo Định luật Amdahl")
        plt.xlabel("Số nhân xử lý (Cores)")
        plt.ylabel("Speedup ($S_p = T_1 / T_p$)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        out_speedup = figures_dir / "amdahl_speedup.png"
        plt.savefig(out_speedup, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Đã lưu biểu đồ Speedup tại: {out_speedup}")

    # 2. Biểu đồ Data Scalability
    scaling_df = df[df["benchmark_type"] == "scaling"].copy()
    if not scaling_df.empty:
        scaling_df["sample_pct"] = scaling_df["sample_fraction"].astype(float) * 100
        scaling_df = scaling_df.sort_values("sample_pct")

        plt.figure(figsize=(7, 5))
        plt.plot(scaling_df["sample_pct"], scaling_df["total_sec"], marker="s", color="#d95f02", linewidth=2)
        plt.title("Khả năng mở rộng theo Quy mô Dữ liệu (Data Scalability)")
        plt.xlabel("Quy mô dữ liệu (% của 32M bản ghi)")
        plt.ylabel("Thời gian thực thi (giây)")
        plt.grid(True, linestyle=":", alpha=0.7)
        out_scaling = figures_dir / "data_scalability.png"
        plt.savefig(out_scaling, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Đã lưu biểu đồ Scalability tại: {out_scaling}")


if __name__ == "__main__":
    plot_benchmark_results()
