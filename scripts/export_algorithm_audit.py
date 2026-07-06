from __future__ import annotations

import argparse

from src.audit.audit_store import AuditStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one Algorithm Audit to CSV.")
    parser.add_argument("--audit-id", default="latest")
    args = parser.parse_args()
    try:
        paths = AuditStore().export_csv(args.audit_id)
    except Exception as exc:
        print(f"导出算法审查失败：{type(exc).__name__}: {exc}")
        return
    print("Algorithm Audit CSV 已导出：")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
