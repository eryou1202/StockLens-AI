from __future__ import annotations

from pathlib import Path

from src.evaluation.rule_revision_lab import RuleRevisionLab


def main() -> None:
    dataset_path = Path("data/ml_dataset.csv")
    output_dir = Path("data/rule_experiments")
    if not dataset_path.exists():
        print("data/ml_dataset.csv 不存在，请先运行：py -m scripts.build_ml_dataset")
        return

    lab = RuleRevisionLab(str(dataset_path))
    try:
        complete = lab.load_complete()
        if complete.empty:
            print("complete 样本为空，暂时无法运行规则实验。")
            return
        lab.print_report()
        lab.export_results(str(output_dir))
    except Exception as exc:
        print(f"规则实验运行失败：{type(exc).__name__}: {exc}")
        return

    print("\nExported:")
    print(output_dir / "rule_revision_summary.csv")
    print(output_dir / "rule_revision_cases.csv")


if __name__ == "__main__":
    main()
