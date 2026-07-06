from __future__ import annotations

import re


class SymbolFormatError(ValueError):
    pass


class SymbolMapper:
    """
    股票代码格式转换器。

    内部统一格式：
    - 000001.SZ
    - 600000.SH
    - 300750.SZ

    Baostock:
    - sz.000001
    - sh.600000

    AKShare:
    - 000001
    - 600000

    TODO:
    - 支持北交所 BJ
    - 支持指数代码映射，例如 000300.SH / sh.000300
    - 支持港股 / 美股
    """

    INTERNAL_RE = re.compile(r"^(\d{6})\.(SZ|SH|BJ)$", re.IGNORECASE)
    BAOSTOCK_RE = re.compile(r"^(sz|sh|bj)\.(\d{6})$", re.IGNORECASE)

    @classmethod
    def normalize(cls, symbol: str) -> str:
        """
        将常见输入统一成内部格式。

        支持：
        - 000001.SZ
        - sz.000001
        - 000001
        """
        symbol = symbol.strip()

        m = cls.INTERNAL_RE.match(symbol)
        if m:
            return f"{m.group(1)}.{m.group(2).upper()}"

        m = cls.BAOSTOCK_RE.match(symbol)
        if m:
            exchange = m.group(1).upper()
            code = m.group(2)
            return f"{code}.{exchange}"

        if re.fullmatch(r"\d{6}", symbol):
            return cls.guess_exchange(symbol)

        raise SymbolFormatError(f"无法识别股票代码格式: {symbol}")

    @classmethod
    def guess_exchange(cls, code: str) -> str:
        """
        根据 A 股常见代码规则猜交易所。

        规则不保证完整，初版够用。
        TODO:
        - 使用股票基础信息表做准确映射
        """
        if code.startswith(("0", "2", "3")):
            return f"{code}.SZ"
        if code.startswith(("5", "6", "9")):
            return f"{code}.SH"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
        raise SymbolFormatError(f"无法根据代码猜交易所: {code}")

    @classmethod
    def to_baostock(cls, symbol: str) -> str:
        internal = cls.normalize(symbol)
        code, exchange = internal.split(".")
        return f"{exchange.lower()}.{code}"

    @classmethod
    def from_baostock(cls, symbol: str) -> str:
        return cls.normalize(symbol)

    @classmethod
    def to_akshare(cls, symbol: str) -> str:
        internal = cls.normalize(symbol)
        code, _exchange = internal.split(".")
        return code

    @classmethod
    def to_tushare(cls, symbol: str) -> str:
        return cls.normalize(symbol)
