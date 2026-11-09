"""Script tự động tải và giải nén bộ dữ liệu MovieLens 32M.

Phụ trách: Nguyễn Đinh Nhật Quang (Data Engineering)
"""

import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

from src.common.config import PROJECT_ROOT, load_config


def download_and_extract(env: str = "dev") -> Path:
    """Tải file ml-32m.zip từ GroupLens và giải nén vào thư mục data/raw/."""
    cfg = load_config(env)
    url = cfg["dataset"]["url"]
    zip_name = cfg["dataset"]["zip_name"]
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / zip_name
    extracted_dir = raw_dir / cfg["dataset"]["extracted_folder"]

    if extracted_dir.exists() and (extracted_dir / "ratings.csv").exists():
        print(f"[INFO] Dữ liệu đã tồn tại tại: {extracted_dir}")
        return extracted_dir

    if not zip_path.exists():
        print(f"[INFO] Đang tải MovieLens 32M từ {url} ...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as file, tqdm(
            desc=zip_name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024 * 1024):
                size = file.write(data)
                bar.update(size)

    print(f"[INFO] Đang giải nén {zip_path} vào {raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    print(f"[INFO] Hoàn tất chuẩn bị dữ liệu thô tại: {extracted_dir}")
    return extracted_dir


if __name__ == "__main__":
    download_and_extract()
