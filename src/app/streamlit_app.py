from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_sell_signals import build_open_position_sell_signals
from scripts.run_recommendations import run_recommendation_analysis
from src.audit.algorithm_audit import AlgorithmAuditRunner
from src.audit.audit_metrics import AuditMetricsBuilder
from src.audit.audit_schema import AuditRequest, AuditSummary
from src.audit.audit_store import AuditStore
from src.audit.universe_loader import load_symbols_from_file, normalize_symbols
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.data.symbol_name_resolver import SymbolNameResolver
from src.portfolio.position_manager import PositionManager
from src.portfolio.position_schema import Position, PositionStatus
from src.recommendation.candidate_pool import CandidatePoolEditor
from src.recommendation.recommendation_explainer import ACTION_LABELS, RecommendationExplainer
from src.recommendation.recommendation_schema import Recommendation, RecommendationAction
from src.scan.a_share_coarse_scanner import AShareCoarseScanner
from src.tracking.recommendation_tracker import RecommendationTracker
from src.tracking.tracking_schema import ManualVerdict


st.set_page_config(page_title="StockLens AI 本地控制台", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.2rem;}
div[data-testid="stMetric"] {background:#f5f7fa;border:1px solid #e6e9ef;padding:12px;border-radius:10px;}
.stAlert {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

SETTINGS = load_settings(PROJECT_ROOT / "config" / "config.yaml")
AI_FILE = PROJECT_ROOT / "data" / "ai_candidates.json"
NAME_RESOLVER = SymbolNameResolver(SETTINGS.database_path, str(AI_FILE))
POOL = CandidatePoolEditor(AI_FILE, NAME_RESOLVER)


def _run_module(module: str, timeout: int = 600) -> str:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, "-m", module], cwd=PROJECT_ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout, check=False)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return output.strip() or f"{module} 执行完成，无文本输出。"


def _recommendations() -> list[Recommendation]:
    return [Recommendation.model_validate(value) for value in st.session_state.get("recommendations", [])]


def _run_recommendations(save: bool = False) -> None:
    with st.spinner("正在读取行情并生成推荐候选..."):
        items = run_recommendation_analysis()
    st.session_state["recommendations"] = [item.model_dump(mode="json") for item in items]
    if save:
        count = RecommendationTracker(SETTINGS.database_path).save_recommendations(items)
        st.success(f"分析完成，新增追踪快照 {count} 条；重复快照已跳过。")
    else:
        st.success("分析完成；本次结果未写入追踪表。")


def _recommendation_rows(items: list[Recommendation]) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        meta = item.metadata
        original_action = meta.get("original_action")
        try:
            original_label = ACTION_LABELS[RecommendationAction(original_action)]
        except (TypeError, ValueError, KeyError):
            original_label = str(original_action or "-")
        rows.append({
            "股票代码": item.symbol,
            "股票名称": item.stock_name or "未知名称",
            "source_type": item.source_type,
            "数据类型": "实时行情" if meta.get("is_realtime") else "非实时 / 昨日收盘数据",
            "实时价": "-" if meta.get("realtime_price") is None else f"{meta['realtime_price']:.2f}",
            "实时涨跌幅": _pct(meta.get("realtime_pct_change")),
            "盘中确认": "是" if meta.get("intraday_confirmed") else "否",
            "原始动作": original_label,
            "当前动作": ACTION_LABELS[item.action],
            "action_level": item.action_level.value,
            "confidence": f"{item.confidence:.2%}",
            "ai_view": item.ai_view,
            "quant_decision": item.quant_decision,
            "final_score": "-" if item.final_score is None else f"{item.final_score:.2f}",
            "数据时间": meta.get("price_time") or "未获取",
            "结论与原因": "；".join(item.reason),
            "风险点": "；".join(item.risks),
        })
    return rows


def _position_rows(items: list[Position]) -> list[dict]:
    return [{
        "id": item.id, "股票代码": item.symbol, "股票名称": item.stock_name or "未知名称",
        "status": item.status.value, "entry_date": item.entry_date.date().isoformat(),
        "entry_price": "-" if item.is_watch_only else f"{item.entry_price:.2f}",
        "position_size": "-" if item.position_size is None else f"{item.position_size:g}",
        "stop_loss": "-" if item.stop_loss_price is None else f"{item.stop_loss_price:.2f}",
        "take_profit": "-" if item.take_profit_price is None else f"{item.take_profit_price:.2f}",
        "notes": item.entry_reason or "-",
    } for item in items]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _tracking_rows(items) -> list[dict]:
    return [{
        "id": item.id, "股票代码": item.symbol, "股票名称": item.stock_name or "未知名称",
        "as_of_time": item.as_of_time.isoformat(), "source_type": item.source_type,
        "action": item.action, "final_score": "-" if item.final_score is None else f"{item.final_score:.2f}",
        "current_price": "-" if item.current_price is None else f"{item.current_price:.2f}",
        "future_return_1d": _pct(item.future_return_1d), "future_return_3d": _pct(item.future_return_3d),
        "future_return_5d": _pct(item.future_return_5d), "future_return_10d": _pct(item.future_return_10d),
        "future_max_drawdown_5d": _pct(item.future_max_drawdown_5d),
        "tracking_status": item.tracking_status.value,
        "manual_verdict": item.manual_verdict.value if item.manual_verdict else "-",
    } for item in items]


def _audit_group_frame(groups: dict, group_name: str) -> pd.DataFrame:
    return pd.DataFrame([
        {group_name: name, **values} for name, values in groups.items()
    ])


def _show_audit_summary(summary: AuditSummary, export_paths: dict | None = None) -> None:
    st.markdown(f"#### 审查结果 · audit_id={summary.audit_id}")
    c1, c2, c3 = st.columns(3)
    c1.metric("样本数", summary.samples_count)
    c2.metric("完整样本", summary.complete_samples)
    c3.metric("排序警告", "是" if summary.ranking_warning else "否")
    st.write("动作分布：", summary.action_distribution)
    st.markdown("##### 按推荐动作分组")
    st.dataframe(
        _audit_group_frame(summary.action_metrics, "action"),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("##### 按量化判断分组")
    st.dataframe(
        _audit_group_frame(summary.quant_decision_metrics, "quant_decision"),
        use_container_width=True,
        hide_index=True,
    )
    correlations = pd.DataFrame([
        {"评分": "综合量化评分", "未来 5 日相关性": summary.score_future_return_corr_5d,
         "未来 10 日相关性": summary.score_future_return_corr_10d},
        {"评分": "最终融合评分", "未来 5 日相关性": summary.final_score_future_return_corr_5d,
         "未来 10 日相关性": summary.final_score_future_return_corr_10d},
    ])
    st.markdown("##### 评分与未来收益相关性")
    st.dataframe(correlations, use_container_width=True, hide_index=True)
    if summary.ranking_warning:
        st.warning("当前样本中积极动作/量化支持的 5 日胜率排序出现反常，请检查规则，不要据此做真实交易判断。")
    if export_paths:
        st.markdown("##### CSV 导出路径")
        st.code("\n".join(f"{name}: {path}" for name, path in export_paths.items()))


st.title("📈 StockLens AI 本地控制台 · MVP v1.2")
st.warning("仅用于研究和辅助决策，不构成投资建议。所有买入、卖出、持有输出均为候选或提醒。")

with st.sidebar:
    st.subheader("运行环境")
    st.code(f"数据库：{SETTINGS.database_path}\n候选池：data/ai_candidates.json")
    st.write(f"行情源：`{SETTINGS.market_provider}`")
    if st.button("快速运行候选推荐", use_container_width=True):
        try:
            _run_recommendations(False)
        except Exception as exc:
            st.error(f"运行失败：{exc}")
    if st.button("快速查看数据状态", use_container_width=True):
        st.session_state["data_output"] = _run_module("scripts.dataset_status")
    st.caption("本地应用不会联网爬新闻、自动交易或连接券商。")

tab_recommend, tab_scan, tab_positions, tab_sell, tab_tracking, tab_data, tab_diagnostics, tab_audit = st.tabs([
    "候选股推荐", "A 股粗扫", "持仓 / 观察管理", "卖出提醒", "追踪复盘", "数据与反馈", "诊断工具", "算法审查"
])

with tab_recommend:
    st.subheader("候选池编辑器")
    try:
        pool_rows = POOL.rows()
        pool_frame = pd.DataFrame(pool_rows).rename(columns={
            "symbol": "股票代码", "stock_name": "股票名称"
        })
        st.dataframe(pool_frame, use_container_width=True, hide_index=True)
    except Exception as exc:
        pool_rows = []
        st.error(f"候选池读取失败：{exc}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 添加人工观察股")
        with st.form("manual_candidate_form"):
            manual_symbol = st.text_input("symbol（必填）", placeholder="300750.SZ")
            manual_name = st.text_input("stock_name（可选）")
            manual_notes = st.text_input("reason / notes（可选）")
            manual_horizons = st.multiselect("expected_horizon_days", [1, 3, 5, 10, 20], default=[3, 5])
            if st.form_submit_button("加入人工观察池", type="primary"):
                try:
                    POOL.add_manual_watch(manual_symbol, manual_name, manual_notes, manual_horizons or [3, 5])
                    st.success("已加入；source_type=manual_watch，不会直接生成 buy_candidate。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"添加失败：{exc}")
    with right:
        st.markdown("#### 删除候选股")
        symbols = [row["symbol"] for row in pool_rows]
        delete_symbol = st.selectbox("按 symbol 删除", symbols, disabled=not symbols)
        st.caption("只修改候选池；不会删除历史信号、positions 或 recommendation_tracking。")
        if st.button("删除所选候选股", disabled=not symbols):
            try:
                POOL.remove(delete_symbol)
                st.success(f"已删除 {delete_symbol}。")
                st.rerun()
            except Exception as exc:
                st.error(f"删除失败：{exc}")

    with st.expander("JSON 导入 / 粘贴（保留 v1.0 功能）"):
        uploaded = st.file_uploader("上传 StockLens Signal Package JSON", type=["json"])
        default_json = AI_FILE.read_text(encoding="utf-8") if AI_FILE.exists() else ""
        if uploaded is not None:
            default_json = uploaded.getvalue().decode("utf-8-sig")
        json_text = st.text_area("或粘贴 Signal Package JSON", value=default_json, height=260)
        if st.button("校验并保存为 ai_candidates.json"):
            try:
                POOL.save_json(json_text)
                st.success("Signal Package v1.0 校验通过并保存。")
                st.rerun()
            except Exception as exc:
                st.error(f"JSON 保存失败：{exc}")

    c1, c2 = st.columns(2)
    if c1.button("刷新推荐池（含盘中确认）", type="primary", use_container_width=True):
        try:
            _run_recommendations(False)
        except Exception as exc:
            st.error(f"分析失败：{type(exc).__name__}: {exc}")
    if c2.button("运行并保存追踪", use_container_width=True):
        try:
            _run_recommendations(True)
        except Exception as exc:
            st.error(f"分析或保存失败：{type(exc).__name__}: {exc}")
    recommendations = _recommendations()
    if recommendations:
        st.dataframe(pd.DataFrame(_recommendation_rows(recommendations)), use_container_width=True, hide_index=True)
        for item in recommendations:
            with st.expander(f"{ACTION_LABELS[item.action]} · {item.symbol} {item.stock_name or '未知名称'} · {item.source_type}"):
                st.text(RecommendationExplainer.format_report(item))

with tab_scan:
    st.subheader("A 股量化粗扫")
    st.info(
        "这是使用现有 RuleScorer 的第一层技术粗筛，不使用 ML、ResearchModelRegistry 或研究模型；"
        "结果仅写入 data/scans/，不会自动修改候选池、持仓、追踪或反馈。"
    )
    with st.form("a_share_coarse_scan_form"):
        scan_scope = st.radio(
            "扫描范围",
            ["全 A 股 stock-only", "指定 symbols-file"],
            horizontal=True,
        )
        scan_symbols_file = st.text_input(
            "symbols-file",
            value="data/universe/liquid100.txt",
            disabled=scan_scope == "全 A 股 stock-only",
        )
        s1, s2, s3 = st.columns(3)
        scan_max_symbols = s1.number_input("max-symbols（0 表示不限）", min_value=0, value=0, step=10)
        scan_limit = s2.number_input("输出数量 limit", min_value=1, value=50, step=5)
        scan_min_amount = s3.number_input(
            "最低 20 日平均成交额",
            min_value=0.0,
            value=30_000_000.0,
            step=5_000_000.0,
            format="%.0f",
        )
        scan_include_risky = st.checkbox("显示高风险候选", value=False)
        run_scan = st.form_submit_button("开始粗扫", type="primary")

    if run_scan:
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def update_scan_progress(symbol, completed, total, status):
            progress_bar.progress(completed / max(total, 1))
            progress_text.caption(f"{completed}/{total} · {symbol} · {status}")

        try:
            scanner = AShareCoarseScanner(SETTINGS, progress_callback=update_scan_progress)
            with st.spinner("正在执行规则量化粗扫；全 A 股首次扫描可能耗时较久..."):
                scan_result = scanner.run(
                    symbols_file=(
                        scan_symbols_file.strip()
                        if scan_scope == "指定 symbols-file" and scan_symbols_file.strip()
                        else None
                    ),
                    max_symbols=int(scan_max_symbols) or None,
                    limit=int(scan_limit),
                    min_avg_amount_20d=float(scan_min_amount),
                    include_risky=scan_include_risky,
                )
                scan_paths = scanner.save_results(scan_result, "data/scans", True, True)
            st.session_state["coarse_scan_result"] = {
                key: value for key, value in scan_result.items() if key != "all_results"
            }
            st.session_state["coarse_scan_paths"] = scan_paths
            st.success("A 股规则量化粗扫完成；未使用 ML，也未自动写入候选池。")
        except Exception as exc:
            st.error(f"粗扫失败：{type(exc).__name__}: {exc}")

    if st.button("读取最近一次粗扫结果", use_container_width=True):
        try:
            st.session_state["coarse_scan_result"] = AShareCoarseScanner.load_latest("data/scans")
            st.session_state["coarse_scan_paths"] = {
                "latest_json": "data/scans/a_share_coarse_scan_latest.json",
                "latest_csv": "data/scans/a_share_coarse_scan_latest.csv",
            }
        except Exception as exc:
            st.error(f"读取失败：{type(exc).__name__}: {exc}")

    scan_result = st.session_state.get("coarse_scan_result")
    if scan_result:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("股票池", scan_result.get("universe_count", 0))
        m2.metric("已完成量化", scan_result.get("scanned_count", 0))
        m3.metric("已排除", scan_result.get("excluded_count", 0))
        m4.metric("最新交易日", scan_result.get("latest_as_of_date") or "-")

        def scan_frame(values):
            rows = []
            for item in values or []:
                rows.append({
                    "rank": item.get("rank"),
                    "股票": f"{item.get('symbol')} {item.get('stock_name') or '未知名称'}",
                    "当前价": item.get("current_price"),
                    "coarse_score": item.get("coarse_score"),
                    "quant_score": item.get("quant_score"),
                    "quant_decision": item.get("quant_decision"),
                    "近 5 日涨跌": _pct(item.get("return_5d")),
                    "近 20 日涨跌": _pct(item.get("return_20d")),
                    "risk_score": item.get("risk_score"),
                    "overheat_score": item.get("overheat_score"),
                    "5 日成交额倍数": item.get("amount_ratio_5d"),
                    "risk_flags": "；".join(item.get("risk_flags") or []),
                })
            return pd.DataFrame(rows)

        st.markdown("#### Top candidates")
        top_values = scan_result.get("top_candidates") or []
        if top_values:
            st.dataframe(scan_frame(top_values), use_container_width=True, hide_index=True)
        else:
            st.info("当前筛选条件下没有 Top candidates。")

        risk_values = scan_result.get("risk_candidates") or []
        if risk_values:
            st.markdown("#### Risk candidates")
            st.dataframe(scan_frame(risk_values), use_container_width=True, hide_index=True)

        st.markdown("#### 排除原因汇总")
        st.json(scan_result.get("excluded_summary") or {})
        if st.session_state.get("coarse_scan_paths"):
            st.caption("；".join(
                f"{name}: {path}"
                for name, path in st.session_state["coarse_scan_paths"].items()
            ))

        st.markdown("#### 手动加入观察池")
        if top_values:
            top_map = {item["symbol"]: item for item in top_values}
            selected_scan_symbol = st.selectbox("选择粗扫候选", list(top_map))
            selected_scan = top_map[selected_scan_symbol]
            if st.button("将所选股票加入人工观察池"):
                try:
                    POOL.add_manual_watch(
                        selected_scan_symbol,
                        selected_scan.get("stock_name"),
                        f"A 股粗扫观察，coarse_score={selected_scan.get('coarse_score')}",
                        [3, 5, 10],
                        metadata={
                            "source": "a_share_coarse_scan",
                            "candidate_type": "coarse_scan_watch",
                            "coarse_score": selected_scan.get("coarse_score"),
                        },
                    )
                    st.success("已按 manual_watch 加入观察池；不会直接形成买入候选。")
                except Exception as exc:
                    st.error(f"加入观察池失败：{exc}")
        else:
            st.caption("暂无可加入观察池的粗扫候选。")

with tab_positions:
    st.subheader("持仓 / 观察管理")
    manager = PositionManager(SETTINGS.database_path)
    for existing in manager.list_positions("all"):
        if not existing.stock_name or existing.stock_name.strip() in {"", "-", "未知名称"}:
            NAME_RESOLVER.update_position_name_if_missing(existing.symbol)
    opens, watches, closed = (manager.list_positions(value) for value in ("open", "watch_only", "closed"))
    for title, values, empty in (
        ("真实 open 持仓", opens, "当前没有真实持仓。"),
        ("watch_only 观察股", watches, "当前没有观察股。"),
        ("closed 历史持仓", closed, "当前没有已关闭历史。"),
    ):
        st.markdown(f"#### {title}")
        if values:
            st.dataframe(
                pd.DataFrame(_position_rows(values)),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(empty)

    mode = st.radio("添加模式", ["添加真实持仓", "添加观察股"], horizontal=True)
    if mode == "添加真实持仓":
        with st.form("add_real_position"):
            a, b, c = st.columns(3)
            symbol = a.text_input("symbol *", placeholder="300750.SZ")
            entry_price = b.number_input("entry_price *", min_value=0.02, value=1.00)
            stock_name = c.text_input("stock_name")
            d, e, f = st.columns(3)
            entry_date = d.date_input("entry_date")
            size = e.number_input("position_size（0 表示空）", min_value=0.0, value=0.0)
            max_days = f.number_input("max_holding_days（0 表示空）", min_value=0, value=0, step=1)
            g, h = st.columns(2)
            stop = g.number_input("stop_loss_price（0 表示空）", min_value=0.0, value=0.0)
            take = h.number_input("take_profit_price（0 表示空）", min_value=0.0, value=0.0)
            reason = st.text_input("entry_reason")
            if st.form_submit_button("添加真实持仓", type="primary"):
                try:
                    position_id = manager.add_position(Position(
                        symbol=symbol, stock_name=stock_name or None,
                        entry_date=datetime.combine(entry_date, datetime.min.time()), entry_price=entry_price,
                        position_size=size or None, max_holding_days=max_days or None,
                        stop_loss_price=stop or None, take_profit_price=take or None,
                        entry_reason=reason or None, entry_action="manual_app",
                    ))
                    st.success(f"真实持仓已添加，id={position_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"添加失败：{exc}")
    else:
        with st.form("add_watch_position"):
            a, b = st.columns(2)
            symbol = a.text_input("symbol *", placeholder="300750.SZ", key="watch_symbol")
            stock_name = b.text_input("stock_name", key="watch_name")
            notes = st.text_input("reason / notes")
            if st.form_submit_button("添加观察股", type="primary"):
                try:
                    position_id = manager.add_position(Position(
                        symbol=symbol, stock_name=stock_name or None, entry_price=0.01,
                        entry_reason=notes or None, entry_action="manual_watch",
                        status=PositionStatus.WATCH_ONLY, metadata={"source_type": "manual_watch"},
                    ))
                    st.success(f"观察股已添加，id={position_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"添加失败：{exc}")

    close_col, remove_col, cleanup_col = st.columns(3)
    with close_col:
        st.markdown("#### 关闭真实持仓")
        with st.form("close_position"):
            close_symbol = st.selectbox("open symbol", [item.symbol for item in opens], disabled=not opens)
            close_price = st.number_input("exit_price", min_value=0.01, value=1.0)
            close_reason = st.text_input("exit_reason")
            if st.form_submit_button("关闭", disabled=not opens):
                manager.close_position(close_symbol, close_price, datetime.now(), close_reason or None)
                st.rerun()
    with remove_col:
        st.markdown("#### 删除观察股")
        watch_symbol = st.selectbox("watch symbol", [item.symbol for item in watches], disabled=not watches)
        if st.button("删除观察股", disabled=not watches):
            manager.remove_watch(symbol=watch_symbol)
            st.rerun()
    with cleanup_col:
        st.markdown("#### 清理异常测试持仓")
        st.caption("将 entry_price <= 0.01 的 open 记录转换为 watch_only，不直接删除。")
        if st.button("执行转换"):
            count = manager.cleanup_test_positions()
            st.success(f"已转换 {count} 条。")
            st.rerun()

with tab_sell:
    st.subheader("卖出提醒与观察风险")
    st.caption("当前价格优先来自实时行情；若实时接口不可用，会明确标注为“非实时，仅最新日线”。")
    if st.button("检查卖出提醒", type="primary"):
        # Never render a previous check while a new refresh is running or after it fails.
        st.session_state.pop("sell_signals", None)
        st.session_state.pop("sell_signals_checked_at", None)
        try:
            with st.spinner("正在检查 open / watch_only..."):
                signals = build_open_position_sell_signals(
                    use_cache=False,
                    force_refresh=True,
                )
            st.session_state["sell_signals"] = [item.model_dump(mode="json") for item in signals]
            st.session_state["sell_signals_checked_at"] = datetime.now().isoformat(timespec="seconds")
        except Exception as exc:
            st.error(f"检查失败：{type(exc).__name__}: {exc}")
    if st.session_state.get("sell_signals_checked_at"):
        st.caption(f"本次检查时间：{st.session_state['sell_signals_checked_at']}")
    signals = [Recommendation.model_validate(item) for item in st.session_state.get("sell_signals", [])]
    if signals:
        rows = []
        for item in signals:
            meta = item.metadata
            rows.append({
                "股票代码": item.symbol, "股票名称": item.stock_name or "未知名称",
                "position_status": meta.get("position_status"), "is_watch_only": meta.get("is_watch_only"),
                "action": ACTION_LABELS[item.action],
                "current_price": "未获取" if meta.get("current_price") is None else f"{meta['current_price']:.2f}",
                "price_time": meta.get("price_time") or "未获取",
                "price_source": meta.get("price_source") or "未获取",
                "is_realtime": bool(meta.get("is_realtime")),
                "数据类型": "实时行情" if meta.get("is_realtime") else "非实时 / 仅最新日线",
                "entry_price": "-" if meta.get("is_watch_only") else f"{meta['entry_price']:.2f}",
                "unrealized_return_percent": "-" if meta.get("is_watch_only") else _pct(meta.get("unrealized_return")),
                "triggered_rules": "；".join(meta.get("triggered_rule_labels", [])),
                "reason_detail": "；".join(item.reason), "risk_detail": "；".join(item.risks),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif "sell_signals" in st.session_state:
        st.info("当前没有 open 持仓或 watch_only 观察股。")

with tab_tracking:
    st.subheader("推荐追踪复盘")
    tracker = RecommendationTracker(SETTINGS.database_path)
    filters = st.columns(3)
    status_filter = filters[0].selectbox("tracking_status", ["all", "tracking", "complete", "failed"])
    action_filter = filters[1].text_input("action 筛选（留空为全部）")
    symbol_filter = filters[2].text_input("symbol 搜索")
    items = tracker.list_tracking(status_filter)
    missing_tracking_names = [
        item.symbol for item in items
        if not item.stock_name or item.stock_name.strip() in {"", "-", "未知名称"}
    ]
    for missing_symbol in dict.fromkeys(missing_tracking_names):
        NAME_RESOLVER.update_tracking_name_if_missing(missing_symbol)
    if missing_tracking_names:
        items = tracker.list_tracking(status_filter)
    if action_filter:
        items = [item for item in items if item.action == action_filter.strip()]
    if symbol_filter:
        items = [item for item in items if symbol_filter.upper() in item.symbol.upper()]
    if items:
        st.dataframe(
            pd.DataFrame(_tracking_rows(items)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("暂无匹配追踪记录。")
    if st.button("更新追踪表现", type="primary"):
        try:
            provider = create_market_data_provider(SETTINGS.market_provider, SETTINGS.cache_dir, True)
            summary = RecommendationTracker(SETTINGS.database_path, provider, SETTINGS.market_adjust_type).update_future_performance()
            st.success("；".join(f"{key}={value}" for key, value in summary.items()))
            st.rerun()
        except Exception as exc:
            st.error(f"更新失败：{exc}")
    st.markdown("#### 人工复盘标记")
    with st.form("verdict_form"):
        tracking_id = st.number_input("tracking_id", min_value=1, step=1)
        verdict = st.selectbox("verdict", [item.value for item in ManualVerdict])
        notes = st.text_area("notes")
        if st.form_submit_button("保存 verdict"):
            try:
                tracker.mark_verdict(int(tracking_id), verdict, notes or None)
                st.success("人工复盘已保存。")
                st.rerun()
            except Exception as exc:
                st.error(f"保存失败：{exc}")

with tab_data:
    st.subheader("数据与反馈")
    actions = {"更新反馈": "scripts.update_feedback", "构建 ML 数据集": "scripts.build_ml_dataset", "查看 dataset_status": "scripts.dataset_status"}
    for column, (label, module) in zip(st.columns(3), actions.items()):
        if column.button(label, use_container_width=True):
            st.session_state["data_output"] = _run_module(module)
    if st.session_state.get("data_output"):
        st.code(st.session_state["data_output"], language="text")

with tab_diagnostics:
    st.subheader("诊断工具")
    actions = {
        "Model Diagnostics": "scripts.diagnose_ml_dataset",
        "Rule Baseline Evaluation": "scripts.evaluate_rule_baseline",
        "Case Inspector": "scripts.inspect_cases",
        "Rule Revision Lab": "scripts.run_rule_revision_lab",
        "训练 ML Baseline": "scripts.train_ml_baseline",
    }
    for label, module in actions.items():
        if st.button(label, key=f"diag_{module}"):
            st.session_state["diagnostic_output"] = _run_module(module)
    if st.session_state.get("diagnostic_output"):
        st.code(st.session_state["diagnostic_output"], language="text")

with tab_audit:
    st.subheader("算法审查实验室（独立实验数据，不影响正式推荐）")
    st.info(
        "该功能用于大范围、长时间审查量化推荐框架。结果只写入 data/audit/，"
        "不会写入正式追踪、持仓、反馈、信号快照或候选池。"
    )
    st.warning("大范围审查可能耗时较久，建议先使用 10～30 只股票测试。")
    with st.form("algorithm_audit_form"):
        d1, d2 = st.columns(2)
        audit_start = d1.date_input("start_date", value=(datetime.now() - timedelta(days=90)).date())
        audit_end = d2.date_input("end_date", value=(datetime.now() - timedelta(days=10)).date())
        audit_symbols = st.text_area(
            "symbols（支持逗号、空格或换行分隔）",
            value="300750.SZ, 000001.SZ, 600030.SH",
        )
        audit_symbols_file = st.text_input("symbols_file（可选）", placeholder="data/universe/my_symbols.txt")
        a1, a2, a3 = st.columns(3)
        audit_step = a1.number_input("step_days", min_value=1, value=5, step=1)
        audit_lookback = a2.number_input("lookback_days", min_value=30, value=120, step=10)
        audit_max_symbols = a3.number_input("max_symbols（0 表示不限）", min_value=0, value=0, step=1)
        audit_name = st.text_input("audit_name", value="streamlit_audit")
        audit_export = st.checkbox("export_csv", value=True)
        run_audit = st.form_submit_button("运行算法审查", type="primary")

    if run_audit:
        try:
            raw_symbols = [audit_symbols]
            if audit_symbols_file.strip():
                raw_symbols.extend(load_symbols_from_file(audit_symbols_file.strip()))
            symbols = normalize_symbols(raw_symbols)
            request = AuditRequest(
                start_date=datetime.combine(audit_start, time(15, 0)),
                end_date=datetime.combine(audit_end, time(15, 0)),
                symbols=symbols,
                step_days=int(audit_step),
                lookback_days=int(audit_lookback),
                max_symbols=int(audit_max_symbols) or None,
                audit_name=audit_name or None,
            )
            audit_store = AuditStore("data/audit/algorithm_audit.sqlite")
            audit_id = audit_store.create_run(request)
            progress_bar = st.progress(0.0)
            progress_text = st.empty()

            def update_audit_progress(symbol, as_of_time, completed, total):
                progress_bar.progress(completed / max(total, 1))
                progress_text.caption(
                    f"{completed}/{total} · {symbol} · {as_of_time.date().isoformat()}"
                )

            provider = create_market_data_provider(
                SETTINGS.market_provider,
                cache_dir="data/audit/cache",
                use_cache=True,
            )
            with st.spinner("正在运行独立算法审查，请勿关闭页面..."):
                samples = AlgorithmAuditRunner(
                    SETTINGS, provider, progress_callback=update_audit_progress
                ).run(request, audit_id=audit_id)
                audit_store.save_samples(samples)
                metrics = AuditMetricsBuilder.build_summary(samples)
                audit_store.finalize_run(audit_id, metrics)
                paths = audit_store.export_csv(audit_id) if audit_export else {}
                summary = audit_store.load_summary(audit_id)
            st.session_state["audit_summary"] = summary.model_dump(mode="json")
            st.session_state["audit_export_paths"] = paths
            st.success(f"算法审查完成：audit_id={audit_id}")
        except Exception as exc:
            st.error(f"算法审查失败：{type(exc).__name__}: {exc}")

    b1, b2 = st.columns(2)
    if b1.button("查看最近一次算法审查结果", use_container_width=True):
        try:
            recent = AuditStore().load_summary("latest")
            st.session_state["audit_summary"] = recent.model_dump(mode="json")
            st.session_state["audit_export_paths"] = {}
        except Exception as exc:
            st.error(f"读取最近审查失败：{type(exc).__name__}: {exc}")
    if b2.button("导出最近一次算法审查 CSV", use_container_width=True):
        try:
            paths = AuditStore().export_csv("latest")
            recent = AuditStore().load_summary("latest")
            st.session_state["audit_summary"] = recent.model_dump(mode="json")
            st.session_state["audit_export_paths"] = paths
            st.success("最近一次审查已导出。")
        except Exception as exc:
            st.error(f"导出最近审查失败：{type(exc).__name__}: {exc}")

    if st.session_state.get("audit_summary"):
        _show_audit_summary(
            AuditSummary.model_validate(st.session_state["audit_summary"]),
            st.session_state.get("audit_export_paths") or None,
        )
