import argparse
import sys
import os
from src.controller import ApplicationController

def main():
    parser = argparse.ArgumentParser(description="Vectorized Alpha Research Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a backtest for a strategy")
    run_parser.add_argument("--strat", required=True, help="Strategy name (folder name in src/strategies)")
    run_parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL)")
    run_parser.add_argument("--interval", default="1d", help="Data interval (e.g., 1d, 4h, 1h)")
    run_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    run_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    run_parser.add_argument("--grid", action="store_true", help="Run grid search hyperparameter sweep")

    # List strategies
    subparsers.add_parser("list", help="List available strategies")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    controller = ApplicationController()

    if args.command == "run":
        try:
            if args.grid:
                controller.run_grid_search(
                    strat_name=args.strat,
                    ticker=args.ticker,
                    interval=args.interval,
                    start=args.start,
                    end=args.end
                )
            else:
                controller.run_backtest(
                    strat_name=args.strat,
                    ticker=args.ticker,
                    interval=args.interval,
                    start=args.start,
                    end=args.end
                )
        except Exception as e:
            print(f"Error executing backtest: {e}")
            sys.exit(1)

    elif args.command == "list":
        strategies = [d for d in os.listdir("src/strategies") 
                      if os.path.isdir(os.path.join("src/strategies", d)) and d != "__pycache__"]
        print("\nAvailable Strategies:")
        for s in strategies:
            print(f" - {s}")
        print()

if __name__ == "__main__":
    main()
