from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_sell_signals import build_open_position_sell_signals
from scripts.run_recommendations import run_recommendation_analysis
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.data.symbol_name_resolver import SymbolNameResolver
from src.portfolio.position_manager import PositionManager
from src.portfolio.position_schema import Position, PositionStatus
from src.recommendation.candidate_pool import CandidatePoolEditor
from src.recommendation.recommendation_explainer import ACTION_LABELS, RecommendationExplainer
from src.recommendation.recommendation_schema import Recommendation
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
    return [{
        "股票代码": item.symbol, "股票名称": item.stock_name or "未知名称",
        "source_type": item.source_type, "action": ACTION_LABELS[item.action],
        "action_level": item.action_level.value, "confidence": f"{item.confidence:.2%}",
        "ai_view": item.ai_view, "quant_decision": item.quant_decision,
        "final_score": "-" if item.final_score is None else f"{item.final_score:.2f}",
        "结论与原因": "；".join(item.reason), "风险点": "；".join(item.risks),
    } for item in items]


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


st.title("📈 StockLens AI 本地控制台 · MVP v1.1.1")
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

tab_recommend, tab_positions, tab_sell, tab_tracking, tab_data, tab_diagnostics = st.tabs([
    "候选股推荐", "持仓 / 观察管理", "卖出提醒", "追踪复盘", "数据与反馈", "诊断工具"
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
    if c1.button("运行候选股分析", type="primary", use_container_width=True):
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
    if st.button("检查卖出提醒", type="primary"):
        try:
            with st.spinner("正在检查 open / watch_only..."):
                signals = build_open_position_sell_signals()
            st.session_state["sell_signals"] = [item.model_dump(mode="json") for item in signals]
        except Exception as exc:
            st.error(f"检查失败：{type(exc).__name__}: {exc}")
    signals = [Recommendation.model_validate(item) for item in st.session_state.get("sell_signals", [])]
    if signals:
        rows = []
        for item in signals:
            meta = item.metadata
            rows.append({
                "股票代码": item.symbol, "股票名称": item.stock_name or "未知名称",
                "position_status": meta.get("position_status"), "is_watch_only": meta.get("is_watch_only"),
                "action": ACTION_LABELS[item.action], "current_price": "-" if meta.get("current_price") is None else f"{meta['current_price']:.2f}",
                "entry_price": "-" if meta.get("is_watch_only") else f"{meta['entry_price']:.2f}",
                "unrealized_return_percent": _pct(meta.get("unrealized_return")),
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
