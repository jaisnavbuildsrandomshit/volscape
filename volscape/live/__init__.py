"""Live public market data for VOLSCAPE.

`feed.py`, `state.py` and `markets.py` are vendored unchanged from
orderflow-terminal (https://github.com/jaisnavbuildsrandomshit/orderflow-terminal, MIT; see
LICENSE-orderflow). They read Binance Spot public market data (real time, no keys) and Yahoo
Finance quotes (delayed). `hub.py` runs them on a background asyncio thread for Streamlit.

This data is MARKET data, separate from the SYNTHETIC option chains: nothing here feeds the
volatility surface or the risk scenarios.
"""
