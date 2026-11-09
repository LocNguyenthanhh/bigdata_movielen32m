# Kế hoạch Đồ án MovieLens 32M Recommendation (Bản Tinh gọn & Thực chiến)

> **Nhóm thực hiện**: Quang – Tâm – Lộc  
> **Môn học**: Big Data (CO5135) – GV: Thầy Thoại Nam (HPC Lab – HCMUT)  
> **Nguyên tắc thiết kế**: Tối ưu cho người **đang đi làm + học 5–6 môn**, tận dụng tối đa **AI để sinh code**, loại bỏ mọi bẫy tốn thời gian (Docker phức tạp, Web UI, K8s, NoSQL cồng kềnh) để tập trung 100% vào **Tính toán phân tán (Apache Spark) và Đánh giá hiệu năng (Benchmarking / Amdahl's Law)** đúng gu giảng viên.

---

## 1. Phân công công việc (Cân bằng & Tinh gọn)

| Thành viên | Vai trò & Trách nhiệm chính | File / Module phụ trách | Thời lượng ước tính |
| :--- | :--- | :--- | :--- |
| **Quang** | **Dữ liệu & Tiền xử lý (Data Engineering)**<br>• Tải và giải nén dữ liệu MovieLens 32M tự động.<br>• Viết PySpark ETL chuyển CSV sang Parquet (partition theo năm, nén Snappy).<br>• Notebook EDA phân tích phân bố ratings, long-tail, độ thưa (sparsity). | • `src/etl/download_data.py`<br>• `src/etl/csv_to_parquet.py`<br>• `src/common/`<br>• `notebooks/01_eda_movielens.ipynb` | ~3–4 giờ/tuần |
| **Tâm** | **Mô hình hóa & Thuật toán lõi (Spark MLlib)**<br>• Huấn luyện mô hình Collaborative Filtering bằng Spark ALS.<br>• Đánh giá chất lượng mô hình (RMSE, Precision@K, NDCG@K, Cold-start analysis).<br>• Viết hàm sinh Top-20 phim gợi ý (loại bỏ phim đã xem).<br>• *(Bonus)*: Phân cụm phim nhanh bằng K-Means trên vector tiềm ẩn `itemFactors`. | • `src/ml/train_als.py`<br>• `src/ml/evaluate_model.py`<br>• `src/ml/generate_topn.py`<br>• `notebooks/02_clustering_bonus.ipynb` | ~3–5 giờ/tuần |
| **Lộc** | **Đánh giá hiệu năng & Báo cáo (Benchmarking & Lead Report)**<br>• Đo đạc thời gian chạy khi thay đổi số Cores (1, 2, 4 cores) $\rightarrow$ Kiểm chứng **Định luật Amdahl (Speedup & Efficiency)**.<br>• Đo đạc thời gian khi tăng quy mô dữ liệu (5%, 20%, 50%, 100%).<br>• Viết script tự động vẽ biểu đồ báo cáo.<br>• Demo tra cứu gợi ý tối giản (Streamlit 40 dòng hoặc CLI).<br>• **Chủ trì viết Báo cáo (Word/LaTeX), làm Slide (08/11 và 12/12) và điều phối nộp bài**. | • `src/benchmarks/`<br>• `src/demo/app_demo.py`<br>• `reports/slide_intro_3pages.pdf`<br>• `reports/final_report.docx/.pdf` | ~4–5 giờ/tuần |

---

## 2. Quy ước kỹ thuật & Hợp đồng dữ liệu (Chốt cố định để làm song song)

Mỗi người có thể dùng AI để sinh code độc lập mà không bị phụ thuộc vào người khác nhờ các quy ước sau:

### C1. Môi trường thực thi & Chế độ chạy
* **Ngôn ngữ**: Python 3.10 / 3.11 + PySpark 3.5.x.
* **Chế độ chạy**: PySpark chạy trực tiếp trên máy cục bộ với `master("local[*]")` (tận dụng tối đa RAM và CPU của máy), **không cần cài đặt cụm Docker phức tạp**. Nếu máy cá nhân RAM yếu ($\le 8\text{GB}$), nhóm dùng chung **Google Colab (High RAM)** hoặc **Databricks Community** (miễn phí, có sẵn Spark).
* **Quản lý cấu hình**: Dùng YAML (`configs/dev.yaml` với `sample_fraction: 0.05` để code test nhanh trong 10 giây; `configs/prod.yaml` với `sample_fraction: 1.0` để chạy dữ liệu thật).

### C2. Cấu trúc lưu trữ dữ liệu (Data Lake Parquet)
* **Gốc dữ liệu**: `data/lake/` (ghi đè `overwrite`, nén `snappy`).
* `ratings/`: Cột `user_id` (int), `movie_id` (int), `rating` (float), `timestamp` (long), `year` (int, cột partition).
* `movies/`: Cột `movie_id` (int), `title` (string), `genres` (array of string), `release_year` (int).
* `output/`: 
  * `topn_recommendations.parquet`: Bảng `(user_id, movie_id, score, rank)`.
  * `metrics.json`: Chứa RMSE, NDCG, thời gian chạy.
  * `benchmark_results.csv`: Chứa thông số Core, Data size, Execution time.

---

## 3. Lộ trình thực hiện 5 tuần (Từ 03/11 đến 12/12/2026)

```
Tuần 1 (03/11 - 08/11)  ──>  [Mốc 1: Nộp 3 Slide Đề tài (08/11) + Xong ETL Dữ liệu]
Tuần 2 (09/11 - 15/11)  ──>  [Mốc 2: Xong Thuật toán ALS & Top-N trên mẫu 5%]
Tuần 3 (16/11 - 22/11)  ──>  [Mốc 3: Chạy huấn luyện trên Full 32M dòng]
Tuần 4 (23/11 - 29/11)  ──>  [Mốc 4: Chạy xong bài đo Benchmark Speedup & Scaling]
Tuần 5 (30/11 - 06/12)  ──>  [Mốc 5: Hoàn thiện Báo cáo, quay Video 10p, Nộp bài (12/12)]
```

### Chi tiết từng tuần:

#### 📍 Tuần 1 (03/11 – 08/11): Đăng ký đề tài & Chuẩn bị dữ liệu
* **Cả nhóm**: Chốt tên đề tài: *"Nghiên cứu, Xây dựng Hệ thống Gợi ý Phim trên Tập Dữ liệu Lớn MovieLens 32M và Đánh giá Hiệu năng Phân tán trên Apache Spark"*.
* **Lộc**: Dùng AI soạn **3 slides giới thiệu đề tài** nộp hạn **08/11/2026** (Slide 1: Topic, Slide 2: Dataset MovieLens 32M, Slide 3: Problems, Architecture & Benchmarking Goals).
* **Quang**: Viết script tải dữ liệu, viết PySpark ETL chuyển CSV sang Parquet phân hoạch (`ratings_parquet`). Chạy thử trên mẫu 5% để cấp cho Tâm.
* **Tâm**: Cài đặt PySpark, chạy thử nghiệm một script mẫu nhỏ để kiểm tra môi trường.

#### 📍 Tuần 2 (09/11 – 15/11): Xây dựng mô hình lõi trên mẫu Dev (5%)
* **Tâm**: Dùng AI viết `train_als.py` và `generate_topn.py`. Chạy thử trên mẫu 5% (`dev.yaml`), kiểm tra hàm tính RMSE và lọc phim đã xem.
* **Quang**: Hoàn thiện notebook `01_eda_movielens.ipynb` (vẽ biểu đồ phân bố ratings, luật lũy thừa long-tail, độ thưa ma trận $>99.8\%$).
* **Lộc**: Soạn sẵn khung Báo cáo cuối kỳ (Word/LaTeX), viết script khung cho phần Benchmark (`benchmark_speedup.py`).

#### 📍 Tuần 3 (16/11 – 22/11): Mở rộng lên dữ liệu đầy đủ (32M)
* **Tâm & Quang**: Chạy thử nghiệm trên toàn bộ 32M ratings với `configs/prod.yaml`. Bật `checkpointDir` và cấu hình bộ nhớ driver/executor trong `spark_session.py` để tránh lỗi tràn bộ nhớ (OOM Heap Space).
* **Tâm**: Tinh chỉnh siêu tham số (rank, regParam, maxIter) để đạt RMSE tối ưu và trích xuất bảng Top-N recommendations.
* *(Bonus Tâm)*: Chạy thử K-Means trên `itemFactors` (50 chiều) để gom nhóm các phim có cùng phong cách.

#### 📍 Tuần 4 (23/11 – 29/11): Đo đạc hiệu năng phân tán (HPC Benchmark)
* **Lộc**: Chạy bộ thực nghiệm đo lường hiệu năng:
  1. *Speedup vs Cores*: Chạy ETL & ALS lần lượt trên `local[1]`, `local[2]`, `local[4]`. Ghi lại thời gian chạy $T_1, T_2, T_4$ và tính Speedup $S_p = T_1 / T_p$.
  2. *Data Scaling*: Chạy trên 5%, 20%, 50%, 100% dữ liệu để thấy sự tăng trưởng thời gian tuyến tính của Spark.
  3. Chạy script `plot_benchmarks.py` để tự động sinh ra các biểu đồ Amdahl's Law chuẩn chỉnh.
* **Lộc**: Viết file demo `app_demo.py` (Streamlit 40 dòng) nhập User ID $\rightarrow$ hiển thị Top 10 phim gợi ý để quay video.

#### 📍 Tuần 5 (30/11 – 06/12): Đóng gói báo cáo & Nộp sớm (Hạn chót 12/12)
* **Lộc**: Điền toàn bộ số liệu, biểu đồ thực nghiệm vào Báo cáo cuối kỳ (Final Report).
* **Quang & Tâm**: Review kỹ thuật, bổ sung phần giải thích mã nguồn và thuật toán ALS.
* **Cả nhóm**: Quay video demo 7–10 phút (giải thích slide, demo 1 lượt chạy pipeline Spark, mở app demo xem kết quả, phân tích biểu đồ Speedup).
* **Nộp bài trước hạn 12/12/2026** và yên tâm tập trung ôn thi cuối kỳ.

---

## 4. Trọng tâm Đánh giá Hiệu năng (Điểm nhấn ăn điểm tối đa)

Để ghi điểm tuyệt đối với Thầy Thoại Nam (HPC Lab), phần Báo cáo của Lộc sẽ có chương riêng về **"Đánh giá Hiệu năng Tính toán Phân tán"**:

1. **Kiểm chứng Định luật Amdahl**:
   $$\text{Speedup } S_p = \frac{T_1}{T_p}, \quad \text{Efficiency } E_p = \frac{S_p}{p}$$
   * Phân tích phần code có thể song song hóa (ALS matrix factorization) so với phần tuần tự (I/O đọc ghi đĩa).
   * Giải thích tại sao khi tăng số Cores thì thời gian giảm nhưng hiệu suất không đạt $100\%$ do chi phí shuffle dữ liệu qua mạng/bộ nhớ.
2. **Khả năng mở rộng theo kích thước dữ liệu (Data Scalability)**:
   * So sánh thời gian xử lý giữa 1.6M, 6.4M, 16M và 32M dòng.
   * Chứng minh kiến trúc Spark DataFrame và định dạng Parquet giúp hệ thống duy trì hiệu năng ổn định, không bị sụp đổ như các thư viện đơn luồng (Pandas/Scikit-learn).
3. **Độ chính xác thuật toán**:
   * Bảng đối chiếu RMSE giữa các giá trị `rank` (10, 20, 50) và `regParam` (0.01, 0.1).
   * Phân tích hiện tượng Cold-start đối với người dùng mới.

---

## 5. Hướng dẫn sử dụng AI hiệu quả (Tránh bẫy thời gian)

* **Khi code PySpark**: Yêu cầu AI luôn thêm cấu hình bộ nhớ và dọn dẹp biến:
  ```python
  # Mẹo: Nhắc AI dùng StructType tường minh khi đọc CSV, 
  # và luôn gọi checkpoint() trong ALS để tránh StackOverflowError trên DAG sâu.
  ```
* **Khi bị lỗi OOM (Tràn RAM)**: Không cố debug chạy 32M ngay, mà copy thông báo lỗi gửi AI hỏi cách giảm `spark.sql.shuffle.partitions` hoặc giảm `rank` của ALS xuống 10–20.
* **Khi viết Báo cáo**: Đưa toàn bộ file `metrics.json` và `benchmark_results.csv` cho AI, yêu cầu AI viết phân tích nhận xét học thuật theo phong cách HPC/Big Data.
