"""Khởi tạo SparkSession tối ưu cho xử lý dữ liệu lớn MovieLens 32M."""

from pathlib import Path
from typing import Any, Dict, Optional
from pyspark.sql import SparkSession

from src.common.config import PROJECT_ROOT, load_config


def get_spark_session(
    config: Optional[Dict[str, Any]] = None,
    master_override: Optional[str] = None,
    app_name_override: Optional[str] = None,
) -> SparkSession:
    """Tạo và cấu hình SparkSession với KryoSerializer, AQE và Checkpoint.

    Args:
        config: Dictionary cấu hình (nếu None sẽ tự gọi load_config()).
        master_override: Ghi đè cấu hình master (ví dụ: 'local[1]', 'local[4]' khi chạy benchmark).
        app_name_override: Ghi đè tên ứng dụng Spark.

    Returns:
        Đối tượng SparkSession đã cấu hình.
    """
    cfg = config or load_config()
    spark_cfg = cfg.get("spark", {})

    app_name = app_name_override or spark_cfg.get("app_name", "MovieLens32M-RecSys")
    master = master_override or spark_cfg.get("master", "local[*]")
    driver_memory = spark_cfg.get("driver_memory", "4g")
    executor_memory = spark_cfg.get("executor_memory", "4g")
    shuffle_partitions = spark_cfg.get("shuffle_partitions", 16)
    serializer = spark_cfg.get("serializer", "org.apache.spark.serializer.KryoSerializer")
    adaptive_enabled = str(spark_cfg.get("adaptive_enabled", True)).lower()

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.serializer", serializer)
        .config("spark.sql.adaptive.enabled", adaptive_enabled)
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Thiết lập thư mục checkpoint để cắt đồ thị RDD (DAG Lineage) cho thuật toán ALS
    checkpoint_rel = cfg.get("paths", {}).get("checkpoint_dir", "data/checkpoints")
    checkpoint_dir = PROJECT_ROOT / checkpoint_rel
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(checkpoint_dir))

    return spark
