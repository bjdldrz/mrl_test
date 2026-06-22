"""实验输出目录工具。"""

from datetime import datetime
from pathlib import Path
import re


def safe_name(text: str, max_len: int = 80) -> str:
    """将实验标签转换为适合目录名的短字符串。"""
    text = text or "run"
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("._-")
    return (text or "run")[:max_len]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_dir(root, name: str, create: bool = True) -> Path:
    """
    在 root 下创建唯一子目录: <name>_<timestamp>[_N]。

    如果同一秒内重复运行, 自动追加 _2/_3 避免覆盖。
    """
    root = Path(root)
    base = f"{safe_name(name)}_{timestamp()}"
    path = root / base
    idx = 2
    while path.exists():
        path = root / f"{base}_{idx}"
        idx += 1
    if create:
        path.mkdir(parents=True, exist_ok=False)
    return path
