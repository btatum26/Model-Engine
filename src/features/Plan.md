================================================================================
DOCUMENT: PLAN.rm (EXPANDED SPECIFICATION - V3)
PROJECT: Vectorized Alpha Research Engine
TIMEFRAME TARGET: 4-Hour (Swing Trading)
STATUS: APPROVED FOR IMPLEMENTATION
LEAD STRATEGIST: Idea Bot
================================================================================

OVERVIEW:
Build a headless, high-speed, purely vectorized Python backtesting and model-development 
environment. This architecture enforces strict separation of concerns, ensuring signal 
generation is completely isolated from trade management and state.

--------------------------------------------------------------------------------
PHASE 1: THE DATA LAKE (Local Cache & High-Speed Ingestion) -> [IMPLEMENTED]
--------------------------------------------------------------------------------
- Standardized SQLite database using SQLAlchemy.
- Implemented SQLite Bulk Upserts (`INSERT OR IGNORE`).
- Isolated ingestion script (`sync_data.py`) to bypass yfinance intraday limits. 
- Data Validation: Engine automatically handles missing candles and forward-fills gaps.

--------------------------------------------------------------------------------
PHASE 2: THE DYNAMIC FEATURE REGISTRY (On-Demand Vectorization)
--------------------------------------------------------------------------------
Objective: Create a library of 30+ highly optimized technical indicators that are 
ONLY calculated when explicitly requested by a strategy's `feature_config`.
- Core Mechanism: Dedicated `features.py` module acting as a function registry.
- Lazy Evaluation: The engine parses the strategy's requested features and dynamically 
  applies only those specific vectorized functions to the OHLCV DataFrame.
- Feature Availability Scope (30+ Supported Options):
    * Price Action & Pivot Geometry (Vectorized Support/Resistance)
    * Volatility Metrics (ATR, Bollinger Width, Keltner Channels)
    * Momentum Oscillators (RSI, MACD, Stochastic, ROC)
    * Trend Alignment (Multiple EMA/SMA distances, ADX)
    * Volume Profiling (VWAP, Volume Z-Scores, OBV)
- Strict Constraint (The T-Zero Rule): Indicators must ONLY be calculated using data 
  available at the close of the current bar. No forward-peeking allowed.
- Output: A lean, custom DataFrame containing OHLCV + ONLY the requested features.

--------------------------------------------------------------------------------
PHASE 3: THE ALPHA ENGINE (Signal Generation & Grid Search)
--------------------------------------------------------------------------------
Objective: Generate stateless directional opinions and support parameter optimization.
- Core Mechanism: `SignalModel` base class that accepts the dynamically generated DataFrame.
- Logic Vectorization: Rewrite strategy logic using Pandas boolean masking.
- Output Specification: A single Pandas Series named `Signal` aligned to the index:
    * 1.0 = Maximum Conviction Long
    * -1.0 = Maximum Conviction Short
    * 0.0 = Neutral / Flat

--------------------------------------------------------------------------------
PHASE 4: THE VECTORIZED PORTFOLIO SIMULATOR (Execution & Risk)
--------------------------------------------------------------------------------
Objective: Simulate execution, enforce trading reality, and manage risk at matrix speeds.
- Core Mechanism: Matrix multiplication of Signals * Asset Returns.
- Strict Constraint (The T+1 Rule): Signals generated on candle Close execute on the next Open.
- Friction Modeling: Subtract a fixed % from every trade for spread/slippage.
- Risk Management: Implement vectorized Trailing Stops using Pandas `cummax()`.
- Output: A continuous `Equity Curve` Series and a `Trade Ledger` DataFrame.

--------------------------------------------------------------------------------
PHASE 5: THE TEARSHEET (Metrics & Evaluation)
--------------------------------------------------------------------------------
Objective: Grade the strategy's real-world viability mathematically.
- Core Mechanism: Ingest the `Equity Curve` and `Trade Ledger` to calculate CAGR, 
  Max Drawdown, Sharpe/Sortino Ratios, and Win Rate.
- Output Format: Formatted console matrix and lightweight HTML plots.

================================================================================
END OF PLAN.rm
================================================================================