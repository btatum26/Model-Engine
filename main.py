import argparse
import sys
import os
import json
from src.controller import ApplicationController, JobPayload, Timeframe

def main():
    parser = argparse.ArgumentParser(description="Vectorized Alpha Research Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Execute command
    exec_parser = subparsers.add_parser("execute", help="Execute a strategy job")
    exec_parser.add_argument("--strat", required=True, help="Strategy name (folder name in src/strategies)")
    exec_parser.add_argument("--assets", required=True, help="Comma-separated ticker symbols (e.g., AAPL,MSFT)")
    exec_parser.add_argument("--interval", default="1d", help="Data interval (e.g., 1d, 4h, 1h)")
    exec_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    exec_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    exec_parser.add_argument("--mode", default="BACKTEST", choices=["BACKTEST", "TRAIN", "SIGNAL_ONLY"], 
                             help="Execution mode")
    exec_parser.add_argument("--multi-mode", default="BATCH", choices=["BATCH", "PORTFOLIO"],
                             help="Multi-asset execution mode")

    # List strategies
    subparsers.add_parser("list", help="List available strategies")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    controller = ApplicationController()

    if args.command == "execute":
        # Prepare the Job Payload using Pydantic models
        asset_list = [a.strip() for a in args.assets.split(",")]
        
        try:
            payload = JobPayload(
                strategy=args.strat,
                assets=asset_list,
                interval=args.interval,
                timeframe=Timeframe(
                    start=args.start,
                    end=args.end
                ),
                mode=args.mode,
                multi_asset_mode=args.multi_mode
            )
        except Exception as e:
            print(f"\nERROR: Invalid job configuration: {e}")
            sys.exit(1)

        print(f"\n--- Vectorized Alpha Research Engine ---")
        print(f"Job Initialized: {payload.mode} for {payload.strategy}")
        print(f"Assets: {', '.join(payload.assets)}")
        print(f"----------------------------------------")

        try:
            result = controller.execute_job(payload)
            
            # Format output based on mode
            if args.mode == "SIGNAL_ONLY":
                print("\nGenerated Signals:")
                print(json.dumps(result, indent=4))
            elif args.mode == "TRAIN":
                print("\nTraining Complete:")
                print(json.dumps(result, indent=4))
            
        except Exception as e:
            print(f"\nERROR: Job execution failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.command == "list":
        strategies_dir = "src/strategies"
        if os.path.exists(strategies_dir):
            strategies = [d for d in os.listdir(strategies_dir) 
                          if os.path.isdir(os.path.join(strategies_dir, d)) and d != "__pycache__"]
            print("\nAvailable Strategies:")
            for s in strategies:
                print(f" - {s}")
        else:
            print(f"\nERROR: Strategies directory '{strategies_dir}' not found.")
        print()

if __name__ == "__main__":
    main()
