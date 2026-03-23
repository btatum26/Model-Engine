import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, Field

from .data_broker.data_broker import DataBroker
from .workspace import WorkspaceManager
from .backtester import LocalBacktester
from .metrics import Tearsheet
from .optimization.optimizer_core import OptimizerCore

class ExecutionMode(str, Enum):
    TRAIN = "TRAIN"
    BACKTEST = "BACKTEST"
    SIGNAL_ONLY = "SIGNAL_ONLY"

class MultiAssetMode(str, Enum):
    BATCH = "BATCH"
    PORTFOLIO = "PORTFOLIO"

class Timeframe(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None

class JobPayload(BaseModel):
    strategy: str
    assets: List[str]
    interval: str
    timeframe: Timeframe = Field(default_factory=Timeframe)
    mode: ExecutionMode
    multi_asset_mode: MultiAssetMode = MultiAssetMode.BATCH

class SignalModel(ABC):
    """The strict interface for all user strategies."""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        Must return a vectorized Pandas Series (float64) 
        ranging from -1.0 (Short) to 1.0 (Long).
        """
        pass

class ApplicationController:
    """
    The Master Orchestrator.
    Wires together DataBroker, WorkspaceManager, Backtester, and Optimizer.
    """
    def __init__(self, strategies_dir: str = "src/strategies"):
        self.strategies_dir = strategies_dir
        self.broker = DataBroker()

    def execute_job(self, payload: Union[dict, JobPayload]) -> Any:
        """
        Single entry point for all engine operations.
        Validates the payload (via Pydantic) and routes to the appropriate pipeline.
        """
        # Validation & Extraction
        if isinstance(payload, dict):
            payload = JobPayload(**payload)
            
        strategy_name = payload.strategy
        assets = payload.assets
        interval = payload.interval
        mode = payload.mode
        multi_asset_mode = payload.multi_asset_mode

        strat_path = os.path.normpath(os.path.join(self.strategies_dir, strategy_name))
        if not os.path.exists(strat_path):
            raise FileNotFoundError(f"Strategy {strategy_name} not found at {strat_path}")

        # Routing Logic
        if mode == ExecutionMode.BACKTEST:
            return self._handle_backtest(
                strat_path, 
                assets, 
                interval, 
                payload.timeframe.start, 
                payload.timeframe.end, 
                multi_asset_mode
            )
        
        elif mode == ExecutionMode.TRAIN:
            return self._handle_train(strat_path, assets, interval, payload.timeframe, payload)
        
        elif mode == ExecutionMode.SIGNAL_ONLY:
            return self._handle_signal_only(
                strat_path, 
                assets, 
                interval, 
                payload.timeframe.start, 
                payload.timeframe.end
            )
        
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _handle_backtest(self, strat_path: str, assets: List[str], interval: str, 
                         start: Optional[str], end: Optional[str], multi_asset_mode: MultiAssetMode):
        """
        Handles BACKTEST pipeline.
        """
        if len(assets) > 1 and multi_asset_mode == MultiAssetMode.PORTFOLIO:
            # Future: Route to PortfolioBacktester
            raise NotImplementedError("PORTFOLIO mode is not yet implemented.")

        # BATCH Mode (Independent runs for each asset)
        all_metrics = {}
        for ticker in assets:
            print(f"\n[BACKTEST] Running {ticker} ({interval})...")
            df_raw = self.broker.get_data(ticker, interval, start, end)
            
            if df_raw.empty:
                print(f"      - WARNING: No data for {ticker}")
                continue

            backtester = LocalBacktester(strat_path)
            signals = backtester.run(df_raw)
            
            metrics = Tearsheet.calculate_metrics(df_raw, signals)
            all_metrics[ticker] = metrics
            
            print(f"      - {ticker} Backtest Complete.")
            Tearsheet.print_summary(metrics)

        return all_metrics

    def _handle_train(self, strat_path: str, assets: List[str], interval: str, 
                      timeframe: Timeframe, payload: JobPayload):
        """
        Handles TRAIN pipeline (Optimization).
        """
        print(f"\n[TRAIN] Initializing Optimizer for {len(assets)} assets...")
        
        # In this version, we use the first asset as the primary dataset reference
        ticker = assets[0]
        dataset_ref = f"{ticker}_{interval}"
        
        # Ensure data is loaded into cache
        self.broker.get_data(ticker, interval, timeframe.start, timeframe.end)

        # Load manifest
        manifest_path = os.path.join(strat_path, "manifest.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Initialize OptimizerCore
        optimizer = OptimizerCore(
            strategy_path=strat_path,
            dataset_ref=dataset_ref,
            manifest=manifest,
            ray_context=None 
        )
        
        return optimizer.run()

    def _handle_signal_only(self, strat_path: str, assets: List[str], interval: str, 
                            start: Optional[str], end: Optional[str]):
        """
        Handles SIGNAL_ONLY pipeline (Inference).
        """
        results = {}
        for ticker in assets:
            print(f"\n[SIGNAL] Generating signal for {ticker}...")
            df_raw = self.broker.get_data(ticker, interval, start, end)
            
            if df_raw.empty:
                results[ticker] = {"error": "No data found"}
                continue

            backtester = LocalBacktester(strat_path)
            signals = backtester.run(df_raw)
            
            # Extract the most recent signal
            last_signal = float(signals.iloc[-1])
            timestamp = signals.index[-1].isoformat()

            results[ticker] = {
                "signal": last_signal,
                "timestamp": timestamp,
                "asset": ticker,
                "mode": "SIGNAL_ONLY"
            }
            print(f"      - {ticker} Signal: {last_signal} at {timestamp}")

        return results
