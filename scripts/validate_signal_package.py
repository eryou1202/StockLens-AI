from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from src.models.signal_package import StockLensSignalPackage


def main() -> None:
    path = Path("data/ai_candidates.json")

    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    try:
        package = StockLensSignalPackage.model_validate(raw)
    except ValidationError as exc:
        print("❌ StockLens Signal Package v1.0 校验失败：")
        print(exc)
        raise SystemExit(1)

    print("✅ StockLens Signal Package v1.0 校验通过")
    print(f"schema_version: {package.schema_version}")
    print(f"as_of_time: {package.run_context.as_of_time}")
    print(f"候选股票数量: {len(package.candidates)}")

    for index, candidate in enumerate(package.candidates, start=1):
        print(
            f"{index}. {candidate.stock_code} {candidate.stock_name} "
            f"events={len(candidate.events)} sources={len(candidate.sources)} "
            f"quant_focus={len(candidate.quant_focus)}"
        )


if __name__ == "__main__":
    main()