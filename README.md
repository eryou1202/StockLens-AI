# AI Candidate Pool + Quant Filter

这是一个“AI 候选股 + 量化曲线分析 + 复盘反馈”的初版项目框架。

这版重点补上了你刚刚强调的内容：

> 量化模型获得股市信息的统一接口。

也就是说，AI 只负责生成候选池；量化模型不直接接触 baostock / akshare，而是统一通过 `MarketDataProvider` 读取行情曲线。

## 核心数据流

```text
AI 信息引擎
    ↓
AI 推荐池 AICandidate[]
    ↓
股票代码 stock_code
    ↓
MarketDataProvider
    ↓
MarketDataBundle
    ↓
QuantEngine
    ↓
趋势 / 曲线 / 风险分析
    ↓
DecisionEngine
    ↓
观察池 / 复盘 / 反馈
```

## 当前包含

```text
src/data/market_data_provider.py       # 行情数据抽象接口
src/data/providers/baostock_provider.py # Baostock 实现骨架
src/data/providers/akshare_provider.py  # AKShare 实现骨架
src/data/symbol_mapper.py              # 股票代码格式转换
src/data/cache_store.py                # 本地缓存接口
src/models/schemas.py                  # 统一数据结构
src/quant/quant_engine.py              # 量化曲线分析接口
src/ai/ai_info_engine.py               # AI 候选池接口
src/decision/decision_engine.py        # AI + 量化融合判断层
src/feedback/feedback_engine.py        # 反馈层接口
```

## 安装

建议 Python 3.11+。

```bash
cd ai_quant_stock_assistant_data_scaffold
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

如果要使用真实 Baostock：

```bash
pip install baostock
```

如果要使用真实 AKShare：

```bash
pip install akshare
```

## 先跑 Mock Demo

```bash
python scripts/run_demo.py
```

这个 demo 不依赖真实数据源，会使用随机行情，主要用于验证系统流程。

## 测试 Baostock Provider

安装 baostock 后：

```bash
python scripts/test_baostock_provider.py
```

## 测试 AKShare Provider

安装 akshare 后：

```bash
python scripts/test_akshare_provider.py
```

## 第一版开发顺序

### V0：跑通项目

- [ ] `python scripts/run_demo.py`
- [ ] 看懂 `src/models/schemas.py`
- [ ] 看懂 `src/data/market_data_provider.py`

### V1：接入真实行情

- [ ] 测试 `BaostockProvider.get_bars()`
- [ ] 测试 `AKShareProvider.get_bars()`
- [ ] 比较两个数据源字段差异
- [ ] 确认复权类型：none / qfq / hfq
- [ ] 加入缓存，避免每次重复请求

### V2：接入量化分析

- [ ] 使用 `MarketDataBundle` 计算 MA5 / MA20 / MA60
- [ ] 计算成交量放大倍数
- [ ] 计算波动率
- [ ] 计算最大回撤
- [ ] 计算过热风险

### V3：接入 AI 候选池

- [ ] AI 输出 `AICandidate`
- [ ] 量化只分析候选池里的股票
- [ ] 判断层融合 AI 信息和曲线分析

### V4：复盘反馈

- [ ] 保存每次判断快照
- [ ] 回填未来 1 / 3 / 5 / 10 日涨跌
- [ ] 统计 AI-only 和 AI+Quant 的差异

## 重要原则

1. 量化模型只读统一后的 `MarketDataBundle`。
2. 不要让 `QuantEngine` 直接 import baostock / akshare。
3. 所有数据都必须带 `as_of_time`，避免未来函数。
4. 第一版先做日线，分钟级后面再加。
5. 先做观察池和复盘，不做自动下单。

## 免责声明

这是学习和研究项目，不构成投资建议，不应用于自动交易或实盘下单。
