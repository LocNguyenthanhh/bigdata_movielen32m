# Slide báo cáo cuối kỳ

**Hệ thống Gợi ý Phim Quy mô lớn với Spark MLlib ALS & Đánh giá Hiệu năng Phân tán trên MovieLens 32M.**
Khoảng 10 phút, quay thành một video; mỗi thành viên tự quay phần của mình. Beamer 16:9, giao diện
Boadilla màu HCMUT, logo BK góc phải trên.

Trình bày theo thứ tự, mỗi người một lượt:

| Người trình bày | File | Nội dung | Thời lượng |
|---|---|---|---|
| A, Nguyễn Đinh Nhật Quang | `main.tex` (tiêu đề, mục lục), `parts/a-data.tex` | Bài toán, dữ liệu, kiến trúc, ETL, EDA | 3:00 |
| B, Nguyễn Văn Tâm | `parts/b-model.tex` | ALS, tuning (E1), Top-N và lọc phim ít đánh giá, chất lượng (E2), cold-start, ví dụ gợi ý, K-Means | 4:00 |
| C, Nguyễn Thanh Lộc | `parts/c-benchmark.tex`, `main.tex` (kết luận) | Data Scaling, Speedup/Amdahl, demo; kết luận và hỏi đáp | 3:00 + 0:45 |

Mỗi người chỉ sửa file phần của mình để ghép không bị xung đột. Trang tiêu đề, mục lục và phần kết
nằm trong `main.tex`.

## Build

```sh
make        # main.pdf (latexmk + XeLaTeX, font TeX Gyre Heros)
make watch  # tự build lại khi lưu
make todo   # số chỗ \todo còn lại trong từng file
```

## Quy ước

- Giao diện và macro (`\todo`, `\partslide`, `\figph`, `\supervisor`) nằm trong `preamble.tex`.
- Hình được tìm trong `../figures/` (biểu đồ do pipeline và notebook sinh ra) và
  `../miniproject-report/images/` (logo, sơ đồ kiến trúc `pipeline.tex` dùng chung với báo cáo).
  Không chép hình sang thư mục này.
- Mọi con số trên slide lấy từ `data/output/` (`metrics.json`, `tuning_results.csv`,
  `benchmark_results.csv`) và phải khớp với báo cáo.
- `\todo{...}` đánh dấu phần còn thiếu, in màu đỏ. `make todo` phải về 0 trước khi quay video.
- Ghi chú cho người nói đặt trong `\note{...}`, bắt đầu bằng thời lượng của khung.
