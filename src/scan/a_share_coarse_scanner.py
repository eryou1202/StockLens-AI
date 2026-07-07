from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

from src.audit.universe_loader import load_symbols_from_file, normalize_symbols
from src.config.settings import AppSettings
from src.data.market_data_provider import MarketDataProvider
from src.data.provider_factory import create_market_data_provider
from src.data.providers.baostock_provider import BaostockProvider
from src.models.schemas import MarketDataBundle
from src.quant.feature_builder import QuantFeatureBuilder
from src.quant.rule_scorer import RuleScorer, clip
from src.reference.symbol_reference import AShareSymbolReference


ProgressCallback = Callable[[str, int, int, str], None]


class AShareCoarseScanner:
    """First-pass rule scan. It never imports or loads an ML model."""

    OUTPUT_FIELDS = [
        "rank", "result_group", "symbol", "stock_name", "exchange", "as_of_date",
        "current_price", "coarse_score", "quant_score", "quant_decision",
        "trend_score", "momentum_score", "volume_score", "risk_score",
        "overheat_score", "macd_score", "return_1d", "return_3d", "return_5d",
        "return_10d", "return_20d", "volume_ratio_5d", "amount_ratio_5d",
        "volatility_20d", "max_drawdown_20d", "rsi_14", "macd_hist",
        "bollinger_position", "avg_amount_20d", "risk_flags", "exclude_reason",
    ]

    def __init__(
        self,
        settings: AppSettings,
        market_data_provider: MarketDataProvider | None = None,
        reference_path: str = "data/reference/a_share_symbols.csv",
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.provider = market_data_provider or create_market_data_provider(
            settings.market_provider,
            cache_dir="data/scans/cache",
            use_cache=True,
        )
        self.reference = AShareSymbolReference(reference_path=reference_path)
        self.progress_callback = progress_callback
        self.feature_builder = QuantFeatureBuilder()
        self.scorer = RuleScorer()

    def run(
        self,
        symbols_file: str | None = None,
        max_symbols: int | None = None,
        limit: int = 50,
        min_avg_amount_20d: float = 30_000_000.0,
        include_risky: bool = False,
        as_of_time: datetime | None = None,
    ) -> dict[str, Any]:
        scan_time = as_of_time or datetime.now()
        universe = self.load_universe(symbols_file, max_symbols)
        total = len(universe)
        rows: list[dict[str, Any]] = []
        fetch_targets: list[dict[str, Any]] = []
        for item in universe:
            if self._excluded_name(item.get("stock_name")):
                rows.append(self._excluded_row(item, "name_risk_filter"))
            else:
                fetch_targets.append(item)

        start_time = scan_time - timedelta(days=260)
        bundles = self._iter_bundles(fetch_targets, start_time, scan_time)
        for completed, (item, bundle, error) in enumerate(bundles, start=1):
            if error is not None or bundle is None:
                row = self._excluded_row(item, f"market_data_error:{error or 'unknown'}")
                rows.append(row)
                self._progress(item["symbol"], completed, len(fetch_targets), "failed")
                continue
            try:
                rows.append(self._analyze(item, bundle, min_avg_amount_20d))
                status = "ok"
            except Exception as exc:
                rows.append(self._excluded_row(
                    item, f"analysis_error:{type(exc).__name__}:{exc}"
                ))
                status = "failed"
            self._progress(item["symbol"], completed, len(fetch_targets), status)

        dated = [row for row in rows if row.get("as_of_date")]
        latest_as_of_date = max((row["as_of_date"] for row in dated), default=None)
        if latest_as_of_date:
            latest_date = datetime.fromisoformat(latest_as_of_date).date()
            for row in rows:
                if not row.get("as_of_date"):
                    continue
                row_date = datetime.fromisoformat(row["as_of_date"]).date()
                if (latest_date - row_date).days > 5:
                    self._add_exclusion(row, "stale_data")

        eligible = [row for row in rows if not row.get("exclude_reason")]
        median_return_20d = self._median([
            value for row in eligible
            if (value := self._number(row.get("return_20d"))) is not None
        ])
        for row in eligible:
            row["coarse_score"] = self._coarse_score(row, median_return_20d)

        risky = [row for row in eligible if row.get("risk_flags")]
        eligible.sort(key=self._sort_key, reverse=True)
        risky.sort(key=self._sort_key, reverse=True)
        excluded = [row for row in rows if row.get("exclude_reason")]
        excluded.sort(key=lambda row: (row.get("symbol") or ""))

        top_limit = max(1, int(limit))
        top_symbols = {row["symbol"] for row in eligible[:top_limit]}
        for index, row in enumerate(eligible, start=1):
            row["rank"] = index
            if row["symbol"] in top_symbols:
                row["result_group"] = "top_candidate"
            elif row.get("risk_flags"):
                row["result_group"] = "risk_candidate"
            else:
                row["result_group"] = "eligible"
        for row in excluded:
            row["rank"] = None
            row["result_group"] = "excluded"

        top_candidates = eligible[:top_limit]
        risk_candidates = risky[:top_limit] if include_risky else []
        exclusion_counts = Counter(
            reason
            for row in excluded
            for reason in str(row.get("exclude_reason") or "").split(";")
            if reason
        )
        return {
            "scan_time": scan_time.isoformat(timespec="seconds"),
            "latest_as_of_date": latest_as_of_date,
            "universe_count": total,
            "scanned_count": sum(1 for row in rows if row.get("quant_score") is not None),
            "excluded_count": len(excluded),
            "top_candidates": top_candidates,
            "risk_candidates": risk_candidates,
            "excluded_summary": dict(sorted(exclusion_counts.items())),
            "metadata": {
                "scan_type": "a_share_coarse_scan",
                "ml_used": False,
                "research_model_used": False,
                "min_avg_amount_20d": min_avg_amount_20d,
                "limit": limit,
                "include_risky": include_risky,
                "risk_candidates_total": len(risky),
                "eligible_total": len(eligible),
                "symbols_file": symbols_file,
            },
            "all_results": eligible + excluded,
        }

    def load_universe(
        self,
        symbols_file: str | None = None,
        max_symbols: int | None = None,
    ) -> list[dict[str, str]]:
        frame = self.reference.load()
        if frame.empty:
            raise FileNotFoundError(
                "data/reference/a_share_symbols.csv 不存在或为空，请先构建 A 股参考表。"
            )
        frame = frame.loc[frame["asset_type"].fillna("").eq("stock")].copy()
        if symbols_file:
            requested = set(load_symbols_from_file(symbols_file))
            frame = frame.loc[frame["symbol"].isin(requested)]
        frame = frame.drop_duplicates("symbol", keep="last").sort_values("symbol", kind="stable")
        if max_symbols is not None:
            frame = frame.head(max(0, int(max_symbols)))
        return [
            {
                "symbol": str(row["symbol"]),
                "stock_name": self._clean_name(row.get("stock_name")),
                "exchange": str(row.get("exchange") or str(row["symbol"]).split(".")[-1]),
            }
            for row in frame.to_dict("records")
        ]

    def save_results(
        self,
        result: dict[str, Any],
        output_dir: str = "data/scans",
        save_json: bool = True,
        save_csv: bool = True,
    ) -> dict[str, str]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromisoformat(result["scan_time"]).strftime("%Y%m%d_%H%M%S")
        paths: dict[str, str] = {}
        if save_csv:
            frame = pd.DataFrame(result.get("all_results", []), columns=self.OUTPUT_FIELDS)
            if "risk_flags" in frame:
                frame["risk_flags"] = frame["risk_flags"].apply(
                    lambda value: ";".join(value) if isinstance(value, list) else value
                )
            timestamped = directory / f"a_share_coarse_scan_{stamp}.csv"
            latest = directory / "a_share_coarse_scan_latest.csv"
            frame.to_csv(timestamped, index=False, encoding="utf-8-sig")
            frame.to_csv(latest, index=False, encoding="utf-8-sig")
            paths.update(csv=str(timestamped), latest_csv=str(latest))
        if save_json:
            payload = {key: value for key, value in result.items() if key != "all_results"}
            timestamped = directory / f"a_share_coarse_scan_{stamp}.json"
            latest = directory / "a_share_coarse_scan_latest.json"
            text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            timestamped.write_text(text, encoding="utf-8")
            latest.write_text(text, encoding="utf-8")
            paths.update(json=str(timestamped), latest_json=str(latest))
        return paths

    @staticmethod
    def load_latest(output_dir: str = "data/scans") -> dict[str, Any]:
        path = Path(output_dir) / "a_share_coarse_scan_latest.json"
        if not path.exists():
            raise FileNotFoundError("尚无粗扫结果，请先运行扫描。")
        return json.loads(path.read_text(encoding="utf-8"))

    def _iter_bundles(
        self,
        universe: list[dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
    ) -> Iterator[tuple[dict[str, Any], MarketDataBundle | None, str | None]]:
        if isinstance(self.provider, BaostockProvider):
            yield from self._iter_baostock_bundles(universe, start_time, end_time)
            return
        for item in universe:
            try:
                yield item, self.provider.get_bars(
                    item["symbol"], start_time, end_time,
                    self.settings.market_frequency, self.settings.market_adjust_type,
                ), None
            except Exception as exc:
                yield item, None, f"{type(exc).__name__}:{exc}"

    def _iter_baostock_bundles(
        self,
        universe: list[dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
    ) -> Iterator[tuple[dict[str, Any], MarketDataBundle | None, str | None]]:
        import baostock as bs
        from src.data.symbol_mapper import SymbolMapper

        login = bs.login()
        if login.error_code != "0":
            error = f"baostock_login:{login.error_msg}"
            for item in universe:
                yield item, None, error
            return
        provider = self.provider
        try:
            for item in universe:
                symbol = item["symbol"]
                try:
                    cached = None
                    if provider.cache is not None:
                        cached = provider.cache.load_bundle(
                            provider.provider_name, symbol, start_time, end_time,
                            self.settings.market_frequency, self.settings.market_adjust_type,
                        )
                    if self._cache_is_fresh(cached, end_time):
                        yield item, cached, None
                        continue

                    query = bs.query_history_k_data_plus(
                        SymbolMapper.to_baostock(symbol),
                        provider.DEFAULT_FIELDS,
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d"),
                        frequency=provider.FREQ_MAP[self.settings.market_frequency],
                        adjustflag=provider.ADJUST_MAP[self.settings.market_adjust_type],
                    )
                    if query.error_code != "0":
                        raise RuntimeError(query.error_msg)
                    raw_rows = []
                    while query.next():
                        raw_rows.append(query.get_row_data())
                    frame = pd.DataFrame(raw_rows, columns=query.fields)
                    bars = provider._frame_to_bars(
                        frame,
                        symbol,
                        self.settings.market_frequency,
                        self.settings.market_adjust_type,
                        SymbolMapper.to_baostock(symbol),
                    )
                    bundle = MarketDataBundle(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        frequency=self.settings.market_frequency,
                        adjust_type=self.settings.market_adjust_type,
                        provider=provider.provider_name,
                        bars=bars,
                        data_quality={"from_cache": False, "rows": len(bars), "coarse_scan": True},
                    )
                    if provider.cache is not None and bars:
                        provider.cache.save_bundle(bundle)
                    yield item, bundle, None
                except Exception as exc:
                    yield item, None, f"{type(exc).__name__}:{exc}"
        finally:
            bs.logout()

    def _analyze(
        self,
        item: dict[str, Any],
        bundle: MarketDataBundle,
        min_avg_amount_20d: float,
    ) -> dict[str, Any]:
        features = self.feature_builder.build(bundle)
        scores = self.scorer.score(features)
        bars = bundle.sorted_bars()
        latest = bars[-1] if bars else None
        avg_amount = self._avg_amount_20d(bars)
        row = {
            "rank": None,
            "result_group": None,
            "symbol": item["symbol"],
            "stock_name": item.get("stock_name") or "未知名称",
            "exchange": item.get("exchange"),
            "as_of_date": latest.trade_time.date().isoformat() if latest else None,
            "current_price": features.latest_close,
            "coarse_score": None,
            "quant_score": scores.quant_score,
            "quant_decision": scores.quant_decision,
            "trend_score": scores.trend_score,
            "momentum_score": scores.momentum_score,
            "volume_score": scores.volume_score,
            "risk_score": scores.risk_score,
            "overheat_score": scores.overheat_score,
            "macd_score": scores.macd_score,
            "return_1d": features.return_1d,
            "return_3d": features.return_3d,
            "return_5d": features.return_5d,
            "return_10d": features.return_10d,
            "return_20d": features.return_20d,
            "volume_ratio_5d": features.volume_ratio_5d,
            "amount_ratio_5d": features.amount_ratio_5d,
            "volatility_20d": features.volatility_20d,
            "max_drawdown_20d": features.max_drawdown_20d,
            "rsi_14": features.rsi_14,
            "macd_hist": features.macd_hist,
            "bollinger_position": features.bollinger_position,
            "avg_amount_20d": avg_amount,
            "risk_flags": self._risk_flags(features, scores),
            "exclude_reason": "",
        }
        if features.data_rows < 80:
            self._add_exclusion(row, "insufficient_history")
        if features.latest_close is None or features.latest_close <= 0:
            self._add_exclusion(row, "latest_price_invalid")
        if avg_amount is None or avg_amount < min_avg_amount_20d:
            self._add_exclusion(row, "low_liquidity")
            if "low_liquidity" not in row["risk_flags"]:
                row["risk_flags"].append("low_liquidity")
        return row

    @staticmethod
    def _coarse_score(row: dict[str, Any], median_return_20d: float | None) -> float:
        score = float(row.get("quant_score") or 0.0)
        decision = row.get("quant_decision")
        if decision == "support":
            score += 5.0
        elif decision == "reject":
            score -= 8.0
        if float(row.get("trend_score") or 0.0) >= 70.0:
            score += 2.0
        amount_ratio = AShareCoarseScanner._number(row.get("amount_ratio_5d"))
        if amount_ratio is not None and 1.2 <= amount_ratio <= 2.5:
            score += 2.0
        risk = float(row.get("risk_score") or 0.0)
        overheat = float(row.get("overheat_score") or 0.0)
        score -= max(0.0, risk - 65.0) * 0.20
        score -= max(0.0, overheat - 65.0) * 0.20
        return_5d = AShareCoarseScanner._number(row.get("return_5d"))
        return_20d = AShareCoarseScanner._number(row.get("return_20d"))
        if return_5d is not None and return_5d > 0.12:
            score -= 4.0
        if return_20d is not None and return_20d > 0.30:
            score -= 5.0
        if return_20d is not None and median_return_20d is not None:
            relative = return_20d - median_return_20d
            if relative >= 0.03:
                score += 2.0
            elif relative <= -0.03:
                score -= 2.0
        return round(clip(score, 0.0, 100.0), 4)

    @staticmethod
    def _risk_flags(features, scores) -> list[str]:
        flags: list[str] = []
        if scores.risk_score >= 75:
            flags.append("high_risk")
        if scores.overheat_score >= 75:
            flags.append("overheat")
        if features.volatility_20d is not None and features.volatility_20d >= 0.05:
            flags.append("high_volatility")
        if features.max_drawdown_20d is not None and features.max_drawdown_20d <= -0.12:
            flags.append("big_drawdown")
        return flags

    @staticmethod
    def _avg_amount_20d(bars) -> float | None:
        values = [
            value for bar in bars[-20:]
            if (value := AShareCoarseScanner._number(bar.amount)) is not None and value >= 0
        ]
        return None if len(values) < 20 else sum(values) / len(values)

    @staticmethod
    def _cache_is_fresh(bundle: MarketDataBundle | None, end_time: datetime) -> bool:
        if bundle is None or len(bundle.bars) < 80:
            return False
        latest = max(bar.trade_time for bar in bundle.bars)
        return (end_time.date() - latest.date()).days <= 10

    @classmethod
    def _excluded_row(cls, item: dict[str, Any], reason: str) -> dict[str, Any]:
        row = {field: None for field in cls.OUTPUT_FIELDS}
        row.update({
            "symbol": item.get("symbol"),
            "stock_name": item.get("stock_name") or "未知名称",
            "exchange": item.get("exchange"),
            "risk_flags": [],
            "exclude_reason": reason,
        })
        return row

    @staticmethod
    def _add_exclusion(row: dict[str, Any], reason: str) -> None:
        current = [item for item in str(row.get("exclude_reason") or "").split(";") if item]
        if reason not in current:
            current.append(reason)
        row["exclude_reason"] = ";".join(current)

    @staticmethod
    def _excluded_name(name: Any) -> bool:
        text = str(name or "").strip().upper()
        return "ST" in text or "退市" in text or "退" in text

    @staticmethod
    def _clean_name(value: Any) -> str:
        if value is None or pd.isna(value):
            return "未知名称"
        text = str(value).strip()
        return text if text and text not in {"-", "nan", "None"} else "未知名称"

    @staticmethod
    def _sort_key(row: dict[str, Any]) -> tuple[float, float]:
        return float(row.get("coarse_score") or -1.0), float(row.get("quant_score") or -1.0)

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0

    def _progress(self, symbol: str, completed: int, total: int, status: str) -> None:
        if self.progress_callback:
            self.progress_callback(symbol, completed, total, status)
        elif completed == 1 or completed == total or completed % 25 == 0:
            print(f"coarse scan {completed}/{total} | {symbol} | {status}", flush=True)
