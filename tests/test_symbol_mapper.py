from src.data.symbol_mapper import SymbolMapper


def test_symbol_mapper():
    assert SymbolMapper.normalize("000001.SZ") == "000001.SZ"
    assert SymbolMapper.normalize("sz.000001") == "000001.SZ"
    assert SymbolMapper.normalize("600000") == "600000.SH"
    assert SymbolMapper.to_baostock("000001.SZ") == "sz.000001"
    assert SymbolMapper.to_akshare("000001.SZ") == "000001"
