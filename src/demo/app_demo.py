"""Giao diện Streamlit tối giản tra cứu Top-N phim gợi ý theo User ID.

Phụ trách: Nguyễn Thanh Lộc (Demo & Report)
Chạy lệnh: streamlit run src/demo/app_demo.py
"""

import pandas as pd
import streamlit as st

from src.common.config import PROJECT_ROOT, load_config


def main() -> None:
    st.set_page_config(page_title="MovieLens 32M Recommender", layout="wide")
    st.title("🎬 Hệ thống Gợi ý Phim Quy mô lớn (MovieLens 32M - Spark ALS)")
    st.caption("Nhóm 261_01: Nguyễn Thanh Lộc • Nguyễn Đinh Nhật Quang • Nguyễn Văn Tâm | GVHD: GS. TS. Thoại Nam – TS. Lê Thanh Vân")

    cfg = load_config("dev")
    topn_path = PROJECT_ROOT / cfg["paths"]["topn_parquet"]
    movies_path = PROJECT_ROOT / cfg["paths"]["movies_parquet"]

    if not topn_path.exists() or not movies_path.exists():
        st.warning("Chưa có dữ liệu gợi ý Parquet. Vui lòng chạy `python run_pipeline.py --env dev` trước!")
        return

    recs_df = pd.read_parquet(topn_path)
    movies_df = pd.read_parquet(movies_path)

    available_users = sorted(recs_df["user_id"].unique().tolist())
    user_id = int(
        st.number_input(
            "Nhập User ID cần tra cứu gợi ý (ID chưa có lịch sử → gợi ý Cold-start):",
            min_value=0,
            value=int(available_users[0]) if available_users else 0,
            step=1,
        )
    )

    if user_id not in set(available_users):
        popular_path = PROJECT_ROOT / cfg["paths"]["popular_parquet"]
        if not popular_path.exists():
            st.warning("User mới chưa có lịch sử và chưa có bảng Bayesian Popularity. Hãy chạy lại pipeline.")
            return
        top_n = int(cfg["als"].get("top_n", 20))
        popular_df = (
            pd.read_parquet(popular_path)
            .sort_values("rank")
            .head(top_n)
            .merge(movies_df, on="movie_id", how="left")
        )
        st.info(f"User #{user_id} chưa có lịch sử đánh giá → gợi ý dự phòng theo Bayesian Popularity (Cold-start).")
        st.dataframe(
            popular_df[["rank", "movie_id", "title", "release_year", "genres", "num_ratings", "avg_rating", "bayes_score"]],
            use_container_width=True,
            hide_index=True,
        )
        return

    user_recs = (
        recs_df[recs_df["user_id"] == user_id]
        .merge(movies_df, on="movie_id", how="left")
        .sort_values("rank")
    )

    st.subheader(f"Top {len(user_recs)} bộ phim gợi ý cho User #{user_id}")
    st.dataframe(
        user_recs[["rank", "movie_id", "title", "release_year", "genres", "score"]],
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
