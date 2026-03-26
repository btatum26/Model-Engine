import argparse
import sys
import os
import subprocess
import pandas as pd
from src.engine.controller import ApplicationController, JobPayload, ExecutionMode
from src.logger import logger
from src.exceptions import EngineError

def main():
    parser = argparse.ArgumentParser(description="Research Engine CLI")
    parser.add_argument("mode", nargs="?", choices=["BACKTEST", "TRAIN", "SIGNAL"], help="Execution mode")
    parser.add_argument("--strategy", help="Strategy folder name")
    parser.add_argument("--ticker", help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--interval", default="1h", help="Data interval (e.g., 1h, 1d)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--gui", action="store_true", help="Launch the Graphical User Interface")

    args = parser.parse_args()

    if args.gui:
        logger.info("Launching GUI...")
        gui_path = os.path.join("src", "gui_launcher.py")
        subprocess.Popen([sys.executable, gui_path])
        return

    if not args.mode:
        parser.print_help()
        return

    # Ensure required arguments are present for CLI modes
    if not args.strategy or not args.ticker:
        print("Error: --strategy and --ticker are required for CLI modes.")
        sys.exit(1)

    # Map CLI modes to ExecutionMode enums
    mode_map = {
        "BACKTEST": ExecutionMode.BACKTEST,
        "TRAIN": ExecutionMode.TRAIN,
        "SIGNAL": ExecutionMode.SIGNAL_ONLY
    }

    controller = ApplicationController()
    
    payload = {
        "strategy": args.strategy,
        "assets": [args.ticker],
        "interval": args.interval,
        "mode": mode_map[args.mode],
        "timeframe": {
            "start": args.start,
            "end": args.end
        }
    }

    try:
        logger.info(f"Starting {args.mode} for {args.ticker} using {args.strategy}")
        result = controller.execute_job(payload)
        
        # Display results based on mode
        if args.mode == "BACKTEST":
            print("\n--- Backtest Results ---")
            for ticker, metrics in result.items():
                print(f"\nAsset: {ticker}")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
        
        elif args.mode == "TRAIN":
            print("\n--- Optimization Results ---")
            print(result)
            
        elif args.mode == "SIGNAL":
            print(f"\n--- Signal Results ---")
            for ticker, data in result.items():
                if "error" in data:
                    print(f"  {ticker}: Error - {data['error']}")
                else:
                    print(f"  {ticker}: {data['signal']} at {data['timestamp']}")

    except EngineError as e:
        logger.error(f"Engine execution failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
