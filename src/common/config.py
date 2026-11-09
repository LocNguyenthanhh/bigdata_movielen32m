"""Module đọc cấu hình YAML (base.yaml + dev.yaml / prod.yaml)."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Gộp đệ quy dictionary override vào base."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(env: str = None) -> Dict[str, Any]:
    """Đọc cấu hình từ configs/base.yaml kết hợp với configs/{env}.yaml.

    Args:
        env: Môi trường chạy ('dev' hoặc 'prod'). Mặc định lấy từ biến môi trường
             APP_ENV hoặc 'dev'.

    Returns:
        Dictionary cấu hình đã hợp nhất.
    """
    env_name = env or os.getenv("APP_ENV", "dev")
    configs_dir = PROJECT_ROOT / "configs"

    base_path = configs_dir / "base.yaml"
    env_path = configs_dir / f"{env_name}.yaml"

    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            env_cfg = yaml.safe_load(f) or {}
        return _deep_merge(base_cfg, env_cfg)

    return base_cfg
