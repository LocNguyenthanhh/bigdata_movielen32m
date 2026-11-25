# Báo cáo Bài tập lớn – CO5135

Báo cáo LaTeX tiếng Việt cho đề tài **Hệ thống Gợi ý Phim Quy mô lớn với Spark MLlib ALS & Đánh giá
Hiệu năng Phân tán trên MovieLens 32M** (Nhóm 261_01). Khổ A4, 12pt, header BK, footer "Trang x/N".

Các chương lý thuyết, dữ liệu và thiết kế đã viết xong. Phần kết quả thực nghiệm còn các ghi chú xám nghiêng
`\dy{...}`: chúng cho biết cần điền gì và lấy số liệu từ đâu. Điền xong thì xoá ghi chú đó.

| File | Phụ trách | Chương |
|---|---|---|
| `sections/tom_tat.tex` | Lộc | Tóm tắt |
| `sections/01_mo_dau.tex` | Lộc | Mở đầu |
| `sections/02_tong_quan.tex` | Quang | Tổng quan và các công trình liên quan |
| `sections/03_co_so_ly_thuyet.tex` | Tâm | Cơ sở lý thuyết (ALS, độ đo, Bayesian Popularity, K-Means, Amdahl) |
| `sections/04_du_lieu_etl.tex` | Quang | Dữ liệu và tiền xử lý phân tán (kiến trúc, EDA, ETL, hợp đồng dữ liệu) |
| `sections/05_mo_hinh_als.tex` | Tâm | Mô hình gợi ý với Spark MLlib ALS |
| `sections/06_danh_gia.tex` | Lộc (E3, E4), Tâm (E1, E2) | Thực nghiệm và đánh giá |
| `sections/07_ket_luan.tex` | Lộc | Kết luận và hướng phát triển |
| `sections/phu_luc.tex` | Lộc | Phụ lục: tái lập thực nghiệm, phân công |

Các file dùng chung (`main.tex`, `Bibliocosas/`, `References/refs.bib`, `images/pipeline.tex`) chỉ sửa khi cả nhóm thống nhất.

## Build

```sh
make           # main.pdf bằng XeLaTeX qua latexmk (MiKTeX hoặc TeX Live)
make watch     # tự build lại khi lưu
make todo      # đếm số ghi chú \dy còn lại trong từng chương
```

Hình được tìm theo tên trong `images/` và `../figures/` (tức `reports/figures/`, nơi
`src/benchmarks/plot_benchmarks.py` và notebook ghi biểu đồ). Hình chưa có sẽ hiện khung xám
(macro `\figph{ten_hinh.png}`); chạy pipeline xong là hình tự xuất hiện.

Sơ đồ kiến trúc `images/pipeline.tex` vẽ bằng TikZ theo đúng pipeline trong mã nguồn và được dùng chung với slide.

## Quy ước

- Viết tiếng Việt. Lần đầu dùng thuật ngữ: in đậm tiếng Việt rồi ghi tiếng Anh, ví dụ `\textbf{Lọc cộng tác} (collaborative filtering)`.
- Tên hàm, cột, lệnh, đường dẫn giữ tiếng Anh trong `\texttt{}`.
- Số liệu dùng `\num{}` (dấu phẩy thập phân) và chỉ lấy từ `data/output/` (`metrics.json`,
  `tuning_results.csv`, `benchmark_results.csv`) hoặc kết quả notebook.
- Từ viết tắt khai báo trong `Bibliocosas/glossary.tex`; tài liệu tham khảo trong `References/refs.bib`.
- Mỗi người chỉ sửa chương của mình để ghép không bị xung đột.
- `make todo` phải về 0 trước khi nộp (12/12/2026).
