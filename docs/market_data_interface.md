# MarketDataProvider 设计说明

## 为什么需要统一接口？

不同数据源的股票代码和字段名不同：

```text
内部统一格式：000001.SZ / 600000.SH
Baostock格式：sz.000001 / sh.600000
AKShare格式：000001 / 600000
```

所以量化模型不能直接依赖某个数据源。正确做法：

```text
Baostock / AKShare
        ↓
Provider 统一字段
        ↓
MarketDataBundle
        ↓
QuantEngine
```

## 核心接口

```python
class MarketDataProvider:
    def get_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "qfq",
    ) -> MarketDataBundle:
        pass

    def get_latest_quote(
        self,
        symbol: str,
        as_of_time: datetime,
    ) -> MarketBar | None:
        pass

    def get_index_bars(
        self,
        index_symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "none",
    ) -> MarketDataBundle:
        pass

    def get_trade_calendar(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[datetime]:
        pass

    def get_stock_status(
        self,
        symbol: str,
        as_of_time: datetime,
    ) -> StockStatus:
        pass
```

## 初版必须支持的字段

```text
symbol
trade_time
frequency
adjust_type
open
high
low
close
pre_close
volume
amount
turnover_rate
pct_chg
trade_status
is_st
provider
fetched_at
```

## Baostock 字段映射

Baostock 常用字段：

```text
date
code
open
high
low
close
preclose
volume
amount
adjustflag
turn
tradestatus
pctChg
peTTM
pbMRQ
psTTM
pcfNcfTTM
isST
```

映射：

```text
date          -> trade_time
code          -> symbol
preclose      -> pre_close
turn          -> turnover_rate
pctChg        -> pct_chg
tradestatus   -> trade_status
isST          -> is_st
peTTM         -> pe_ttm
pbMRQ         -> pb
psTTM         -> ps_ttm
```

## AKShare 字段映射

AKShare `stock_zh_a_hist` 常见字段：

```text
日期
股票代码
开盘
收盘
最高
最低
成交量
成交额
振幅
涨跌幅
涨跌额
换手率
```

映射：

```text
日期      -> trade_time
股票代码  -> symbol
开盘      -> open
收盘      -> close
最高      -> high
最低      -> low
成交量    -> volume
成交额    -> amount
涨跌幅    -> pct_chg
换手率    -> turnover_rate
```

## TODO

- [ ] 验证不同数据源的复权方式是否一致
- [ ] 加入指数代码映射
- [ ] 加入行业指数接口
- [ ] 加入交易日历缓存
- [ ] 加入停牌和 ST 过滤
- [ ] 加入数据质量检查
