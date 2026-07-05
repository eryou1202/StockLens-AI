from __future__ import annotations

import json
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
from src.models.signal_package import StockLensSignalPackage
from src.portfolio.position_manager import PositionManager
from src.portfolio.position_schema import Position
from src.recommendation.recommendation_explainer import ACTION_LABELS, RecommendationExplainer
from src.recommendation.recommendation_schema import Recommendation


st.set_page_config(page_title="StockLens AI 本地控制台", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
    div[data-testid="stMetric"] {background:#f5f7fa;border:1px solid #e6e9ef;padding:12px;border-radius:10px;}
    .stAlert {border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

SETTINGS = load_settings(PROJECT_ROOT / "config" / "config.yaml")
AI_FILE = PROJECT_ROOT / "data" / "ai_candidates.json"


def _run_module(module: str, timeout: int = 600) -> str:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return output.strip() or f"{module} 执行完成，无文本输出。"


def _recommendation_rows(items: list[Recommendation]) -> list[dict]:
    return [
        {
            "symbol": item.symbol,
            "stock_name": item.stock_name,
            "action": ACTION_LABELS[item.action],
            "action_level": item.action_level.value,
            "confidence": item.confidence,
            "ai_view": item.ai_view,
            "quant_decision": item.quant_decision,
            "final_score": item.final_score,
            "reason": item.reason[0] if item.reason else "-",
        }
        for item in items
    ]


st.title("📈 StockLens AI 本地控制台")
st.warning("StockLens AI 仅用于研究和辅助决策，不构成投资建议。所有买入/卖出输出均为候选或提醒。")

with st.sidebar:
    st.subheader("运行环境")
    st.code(f"数据库：{SETTINGS.database_path}\n候选池：data/ai_candidates.json")
    st.write(f"行情源：`{SETTINGS.market_provider}`")
    st.write(f"AI 候选池：{'✅ 已存在' if AI_FILE.exists() else '❌ 不存在'}")
    st.divider()
    st.subheader("常用操作")
    if st.button("快速运行候选推荐", use_container_width=True):
        try:
            with st.spinner("正在分析..."):
                quick_items = run_recommendation_analysis()
            st.session_state["recommendations"] = [item.model_dump(mode="json") for item in quick_items]
            st.success("推荐已更新，请查看“候选股推荐”页签。")
        except Exception as exc:
            st.error(f"运行失败：{exc}")
    if st.button("快速查看数据状态", use_container_width=True):
        st.session_state["data_output"] = _run_module("scripts.dataset_status")
        st.success("状态已更新，请查看“数据与反馈”页签。")
    st.divider()
    st.caption("信息搜集仍由用户在 ChatGPT 中完成；本地应用不会联网爬新闻或自动交易。")

tab_recommend, tab_positions, tab_sell, tab_data, tab_diagnostics = st.tabs(
    ["候选股推荐", "持仓管理", "卖出提醒", "数据与反馈", "诊断工具"]
)

with tab_recommend:
    st.subheader("候选股推荐")
    uploaded = st.file_uploader("上传 StockLens Signal Package JSON", type=["json"])
    default_json = AI_FILE.read_text(encoding="utf-8") if AI_FILE.exists() else ""
    if uploaded is not None:
        default_json = uploaded.getvalue().decode("utf-8-sig")
    json_text = st.text_area("或粘贴 Signal Package JSON", value=default_json, height=260)
    col_save, col_run = st.columns(2)
    with col_save:
        if st.button("保存为 ai_candidates.json", use_container_width=True):
            try:
                payload = json.loads(json_text)
                StockLensSignalPackage.model_validate(payload)
                AI_FILE.parent.mkdir(parents=True, exist_ok=True)
                AI_FILE.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.success("候选池已校验并保存。")
            except Exception as exc:
                st.error(f"JSON 保存失败：{exc}")
    with col_run:
        if st.button("运行候选股分析", type="primary", use_container_width=True):
            try:
                with st.spinner("正在读取行情并生成推荐候选..."):
                    items = run_recommendation_analysis()
                st.session_state["recommendations"] = [item.model_dump(mode="json") for item in items]
            except Exception as exc:
                st.error(f"候选股分析失败：{type(exc).__name__}: {exc}")

    recommendation_payloads = st.session_state.get("recommendations", [])
    recommendations = [Recommendation.model_validate(item) for item in recommendation_payloads]
    if recommendations:
        st.dataframe(pd.DataFrame(_recommendation_rows(recommendations)), use_container_width=True, hide_index=True)
        for item in recommendations:
            with st.expander(f"{ACTION_LABELS[item.action]} · {item.symbol} {item.stock_name or ''}"):
                st.text(RecommendationExplainer.format_report(item))

with tab_positions:
    st.subheader("持仓管理")
    manager = PositionManager(SETTINGS.database_path)
    open_positions = manager.list_positions("open")
    if open_positions:
        st.dataframe(
            pd.DataFrame([item.model_dump(mode="json") for item in open_positions]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("当前没有 open 持仓。")

    st.markdown("#### 添加持仓")
    with st.form("add_position_form"):
        c1, c2, c3 = st.columns(3)
        symbol = c1.text_input("symbol", placeholder="300750.SZ")
        stock_name = c2.text_input("stock_name")
        entry_date = c3.date_input("entry_date")
        c4, c5, c6 = st.columns(3)
        entry_price = c4.number_input("entry_price", min_value=0.01, value=1.0)
        position_size = c5.number_input("position_size（0 表示空）", min_value=0.0, value=0.0)
        max_days = c6.number_input("max_holding_days（0 表示空）", min_value=0, value=0, step=1)
        c7, c8 = st.columns(2)
        stop_loss = c7.number_input("stop_loss_price（0 表示空）", min_value=0.0, value=0.0)
        take_profit = c8.number_input("take_profit_price（0 表示空）", min_value=0.0, value=0.0)
        entry_reason = st.text_input("entry_reason")
        submitted = st.form_submit_button("添加持仓", type="primary")
        if submitted:
            try:
                position_id = manager.add_position(
                    Position(
                        symbol=symbol,
                        stock_name=stock_name or None,
                        entry_date=datetime.combine(entry_date, datetime.min.time()),
                        entry_price=entry_price,
                        position_size=position_size or None,
                        entry_reason=entry_reason or None,
                        entry_action="manual_app",
                        stop_loss_price=stop_loss or None,
                        take_profit_price=take_profit or None,
                        max_holding_days=max_days or None,
                    )
                )
                st.success(f"持仓已添加，id={position_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"添加失败：{exc}")

    st.markdown("#### 关闭持仓")
    with st.form("close_position_form"):
        symbols = [item.symbol for item in open_positions]
        close_symbol = st.selectbox("symbol", symbols, disabled=not symbols)
        close_price = st.number_input("exit_price", min_value=0.01, value=1.0)
        close_date = st.date_input("exit_date", key="close_date")
        close_reason = st.text_input("exit_reason")
        close_submitted = st.form_submit_button("关闭持仓", disabled=not symbols)
        if close_submitted:
            try:
                manager.close_position(
                    close_symbol,
                    close_price,
                    datetime.combine(close_date, datetime.min.time()),
                    close_reason or None,
                )
                st.success("持仓已关闭。")
                st.rerun()
            except Exception as exc:
                st.error(f"关闭失败：{exc}")

with tab_sell:
    st.subheader("卖出提醒")
    if st.button("检查卖出提醒", type="primary"):
        try:
            with st.spinner("正在检查 open positions..."):
                signals = build_open_position_sell_signals()
            st.session_state["sell_signals"] = [item.model_dump(mode="json") for item in signals]
        except Exception as exc:
            st.error(f"检查失败：{type(exc).__name__}: {exc}")
    signals = [Recommendation.model_validate(item) for item in st.session_state.get("sell_signals", [])]
    if signals:
        rows = []
        for item in signals:
            rows.append(
                {
                    "symbol": item.symbol,
                    "stock_name": item.stock_name,
                    "action": ACTION_LABELS[item.action],
                    "action_level": item.action_level.value,
                    "current_price": item.metadata.get("current_price"),
                    "entry_price": item.metadata.get("entry_price"),
                    "unrealized_return": item.metadata.get("unrealized_return"),
                    "reason": "；".join(item.reason),
                    "risks": "；".join(item.risks),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif "sell_signals" in st.session_state:
        st.info("当前没有 open 持仓。")

with tab_data:
    st.subheader("数据与反馈")
    data_actions = {
        "更新反馈": "scripts.update_feedback",
        "构建 ML 数据集": "scripts.build_ml_dataset",
        "查看 dataset_status": "scripts.dataset_status",
    }
    columns = st.columns(3)
    for column, (label, module) in zip(columns, data_actions.items()):
        if column.button(label, use_container_width=True):
            with st.spinner(f"正在执行 {label}..."):
                st.session_state["data_output"] = _run_module(module)
    if st.session_state.get("data_output"):
        st.code(st.session_state["data_output"], language="text")

with tab_diagnostics:
    st.subheader("诊断工具")
    diagnostic_actions = {
        "Model Diagnostics": "scripts.diagnose_ml_dataset",
        "Rule Baseline Evaluation": "scripts.evaluate_rule_baseline",
        "Case Inspector": "scripts.inspect_cases",
        "Rule Revision Lab": "scripts.run_rule_revision_lab",
        "训练 ML Baseline": "scripts.train_ml_baseline",
    }
    for label, module in diagnostic_actions.items():
        if st.button(label, key=f"diag_{module}"):
            with st.spinner(f"正在执行 {label}..."):
                st.session_state["diagnostic_output"] = _run_module(module)
    if st.session_state.get("diagnostic_output"):
        st.code(st.session_state["diagnostic_output"], language="text")
