# Research Engine

A modular platform for financial strategy development, backtesting, and optimization.

## Core Components

### 1. Data Management (`src/data_broker/`)
- **Fetcher**: Interface for retrieving historical and live market data (currently supports yfinance). Handles interval mapping (e.g., 4h resampling) and timezone normalization.
- **DataBroker**: High-level interface that manages data retrieval and local caching for performance.

### 2. Feature Pipeline (`src/features/`)
- **Feature Registry**: Dynamic system for registering and discovering technical indicators and alpha factors.
- **Orchestrator**: Executes batch feature computation.
- **Blast Shield**: Safety mechanism that prevents feature logic from modifying the source data in-place, ensuring data integrity across computations.
- **Cache**: In-memory caching for individual feature series to avoid redundant calculations.

### 3. Strategy Execution (`src/backtester.py`)
- **LocalBacktester**: Loads user-defined strategy models from directory-based workspaces. 
- **Dynamic Context**: Automatically generates a `context.py` for each strategy, mapping human-readable feature names to their respective column names in the computed DataFrame.
- **Grid Search**: Built-in support for parameter sweeps across defined hyperparameter bounds.

### 4. Optimization (`src/optimization/`)
- **OptimizerCore**: Utilizes Optuna for intelligent hyperparameter optimization.
- **CPCV Splitter**: Implements Combinatorial Purged Cross-Validation to prevent data leakage and overfit in financial time-series models.

### 5. Distribution (`src/bundler.py`)
- **Bundler**: Packages strategy code, manifests, and trained artifacts into standalone `.strat` files for deployment.

---

## Tooling & Interfaces

### 1. Command Line Interface (CLI)
Used for direct interaction with the engine for backtesting, training, and signal generation.

```bash
# Launch the Graphical User Interface
uv run python main.py --gui

# Run a backtest
uv run python main.py BACKTEST --strategy momentum_surge --ticker AAPL --interval 1h

# Start an optimization/training job
uv run python main.py TRAIN --strategy momentum_surge --ticker TSLA --interval 1d

# Generate the latest signal only
uv run python main.py SIGNAL --strategy momentum_surge --ticker SPY
```

### 2. Graphical User Interface (GUI)
A Tkinter-based dashboard for strategy management, configuration, and remote job submission.

```bash
# Can be launched via the CLI
uv run python main.py --gui

# Or directly
uv run python src/gui_launcher.py
```
**Features:**
- Dynamic configuration of strategy hyperparameters via `manifest.json`.
- Strategy workspace synchronization (updates `context.py` and local metadata).
- Real-time job status tracking from the Compute Daemon.
- Strategy creation wizard.

### 3. Compute Daemon
A FastAPI-based background service that manages a job queue and executes long-running tasks (like training) in separate processes.

```bash
# Start the daemon
uv run python -m uvicorn src.daemon.main:app --reload --port 8000
```
- **Beacon System**: Automatically writes an `api_beacon.json` to the `transit/` directory so the GUI can discover the daemon's address.
- **Job Registry**: Uses a local SQLite database (`jobs.db`) to persist job history and status.

---

## Development Workflow

1.  **Create Strategy**: Use the GUI or create a new folder in `src/strategies/` with a `manifest.json`.
2.  **Define Features**: Add feature requirements to the manifest.
3.  **Sync Workspace**: Use `WorkspaceManager` (via GUI "Sync" button) to generate the strategy `context.py`.
4.  **Implement Logic**: Write signal generation logic in `model.py` using the auto-generated `context` attributes.
5.  **Test/Optimize**: Use the CLI or GUI to run backtests and optimization sweeps.
6.  **Bundle**: Export the finalized strategy to an artifact for deployment.

## Testing

Run the full test suite to ensure system integrity:
```bash
uv run pytest
```
The suite covers data fetching, backtesting logic, API endpoints, and the feature blast shield.
