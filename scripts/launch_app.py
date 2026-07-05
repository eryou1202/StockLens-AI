from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if importlib.util.find_spec("streamlit") is None:
        print("未安装 streamlit，请运行：pip install streamlit")
        return
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "src" / "app" / "streamlit_app.py"
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=project_root,
            check=False,
        )
    except OSError as exc:
        print(f"Streamlit 启动失败：{exc}")


if __name__ == "__main__":
    main()
