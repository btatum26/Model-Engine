import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import json
import threading
import queue
import sys
from datetime import datetime

# Add project root to sys.path to allow internal imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controller import ApplicationController, ExecutionMode, JobPayload, Timeframe, MultiAssetMode
from src.workspace import WorkspaceManager
from src.bundler import Bundler

class GuiLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Vectorized Alpha Research Engine - Control Panel")
        self.root.geometry("800x900")
        
        self.controller = ApplicationController()
        self.queue = queue.Queue()
        self.strategies_dir = "src/strategies"
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self._setup_logging()
        self._create_widgets()
        self._refresh_strategies()
        self._listen_to_queue()

    def _setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"gui_log_{timestamp}.txt")

    def _log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.console.insert(tk.END, formatted_message)
        self.console.see(tk.END)
        with open(self.log_file, "a") as f:
            f.write(formatted_message)

    def _create_widgets(self):
        # --- Main Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Section 1: Strategy Selector ---
        selector_frame = ttk.LabelFrame(main_frame, text="Strategy Selector", padding="5")
        selector_frame.pack(fill=tk.X, pady=5)

        ttk.Label(selector_frame, text="Select Strategy:").pack(side=tk.LEFT, padx=5)
        self.strategy_var = tk.StringVar()
        self.strategy_dropdown = ttk.Combobox(selector_frame, textvariable=self.strategy_var, state="readonly")
        self.strategy_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.strategy_dropdown.bind("<<ComboboxSelected>>", self._on_strategy_selected)

        ttk.Button(selector_frame, text="Refresh", command=self._refresh_strategies).pack(side=tk.LEFT, padx=5)
        ttk.Button(selector_frame, text="Create New Strategy", command=self._create_new_strategy_popup).pack(side=tk.LEFT, padx=5)

        # --- Section 2: Strategy Configurator ---
        self.config_frame = ttk.LabelFrame(main_frame, text="Strategy Configurator", padding="5")
        self.config_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollable area for config
        self.config_canvas = tk.Canvas(self.config_frame)
        self.config_scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=self.config_canvas.yview)
        self.scrollable_config = ttk.Frame(self.config_canvas)

        self.scrollable_config.bind(
            "<Configure>",
            lambda e: self.config_canvas.configure(scrollregion=self.config_canvas.bbox("all"))
        )

        self.config_canvas.create_window((0, 0), window=self.scrollable_config, anchor="nw")
        self.config_canvas.configure(yscrollcommand=self.config_scrollbar.set)

        self.config_canvas.pack(side="left", fill="both", expand=True)
        self.config_scrollbar.pack(side="right", fill="y")

        # --- Section 3: Execution Routing ---
        routing_frame = ttk.LabelFrame(main_frame, text="Execution Routing", padding="5")
        routing_frame.pack(fill=tk.X, pady=5)

        # Common Job Parameters (Ticker, Interval, etc.)
        param_frame = ttk.Frame(routing_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Ticker:").pack(side=tk.LEFT, padx=2)
        self.ticker_var = tk.StringVar(value="AAPL")
        ttk.Entry(param_frame, textvariable=self.ticker_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(param_frame, text="Interval:").pack(side=tk.LEFT, padx=2)
        self.interval_var = tk.StringVar(value="1h")
        self.interval_dropdown = ttk.Combobox(param_frame, textvariable=self.interval_var, values=["1m", "5m", "15m", "1h", "1d"], width=5)
        self.interval_dropdown.pack(side=tk.LEFT, padx=5)

        button_frame = ttk.Frame(routing_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Sync Data", command=self._sync_data).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Run Backtest", command=lambda: self._run_job(ExecutionMode.BACKTEST)).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Run Grid Search", command=lambda: self._run_job(ExecutionMode.TRAIN)).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Bundle Artifact", command=self._bundle_artifact).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # --- Section 4: Progress Tracker ---
        progress_frame = ttk.Frame(main_frame, padding="5")
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

        # --- Section 5: Output Console ---
        console_frame = ttk.LabelFrame(main_frame, text="Output Console", padding="5")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=15, state="normal")
        self.console.pack(fill=tk.BOTH, expand=True)

    def _refresh_strategies(self):
        if not os.path.exists(self.strategies_dir):
            os.makedirs(self.strategies_dir)
        
        strategies = [d for d in os.listdir(self.strategies_dir) if os.path.isdir(os.path.join(self.strategies_dir, d))]
        self.strategy_dropdown['values'] = strategies
        if strategies:
            if not self.strategy_var.get():
                self.strategy_var.set(strategies[0])
            self._on_strategy_selected()

    def _on_strategy_selected(self, event=None):
        strategy = self.strategy_var.get()
        if not strategy:
            return
        
        manifest_path = os.path.join(self.strategies_dir, strategy, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            self._populate_configurator()
        else:
            self._log(f"Warning: manifest.json not found for {strategy}")

    def _populate_configurator(self):
        # Clear existing widgets in scrollable_config
        for widget in self.scrollable_config.winfo_children():
            widget.destroy()

        # Hyperparameters
        ttk.Label(self.scrollable_config, text="Hyperparameters", font=("", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.hp_entries = {}
        hparams = self.manifest.get("hyperparameters", {})
        for key, val in hparams.items():
            frame = ttk.Frame(self.scrollable_config)
            frame.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(frame, text=key, width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(val))
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.hp_entries[key] = var

        # Features (Simplified for now - raw JSON edit for complex ones)
        ttk.Label(self.scrollable_config, text="Features (JSON)", font=("", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.features_text = scrolledtext.ScrolledText(self.scrollable_config, height=5)
        self.features_text.pack(fill=tk.X, padx=10, pady=2)
        self.features_text.insert(tk.END, json.dumps(self.manifest.get("features", []), indent=4))

        # Parameter Bounds (for Optuna/Grid)
        ttk.Label(self.scrollable_config, text="Parameter Bounds (JSON)", font=("", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.bounds_text = scrolledtext.ScrolledText(self.scrollable_config, height=5)
        self.bounds_text.pack(fill=tk.X, padx=10, pady=2)
        self.bounds_text.insert(tk.END, json.dumps(self.manifest.get("parameter_bounds", {}), indent=4))

        ttk.Button(self.scrollable_config, text="Save Manifest", command=self._save_manifest).pack(pady=10)

    def _save_manifest(self):
        strategy = self.strategy_var.get()
        if not strategy:
            return

        try:
            # Reconstruct manifest
            new_hparams = {}
            for key, var in self.hp_entries.items():
                val = var.get()
                # Try to convert to numeric if possible
                try:
                    if '.' in val:
                        new_hparams[key] = float(val)
                    else:
                        new_hparams[key] = int(val)
                except ValueError:
                    new_hparams[key] = val

            new_features = json.loads(self.features_text.get("1.0", tk.END))
            new_bounds = json.loads(self.bounds_text.get("1.0", tk.END))

            self.manifest["hyperparameters"] = new_hparams
            self.manifest["features"] = new_features
            self.manifest["parameter_bounds"] = new_bounds

            manifest_path = os.path.join(self.strategies_dir, strategy, "manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=4)
            
            self._log(f"Manifest saved for {strategy}")
            messagebox.showinfo("Success", f"Manifest for {strategy} updated.")
        except Exception as e:
            self._log(f"Error saving manifest: {e}")
            messagebox.showerror("Error", f"Failed to save manifest: {e}")

    def _sync_data(self):
        strategy = self.strategy_var.get()
        if not strategy: return
        
        self._save_manifest() # Ensure we sync latest UI values
        
        def task():
            try:
                self.queue.put(("LOG", f"Syncing {strategy}..."))
                self.queue.put(("PROGRESS", 20))
                
                strat_path = os.path.join(self.strategies_dir, strategy)
                wm = WorkspaceManager(strat_path)
                wm.refresh() # WorkspaceManager.refresh() generates context.py
                
                self.queue.put(("PROGRESS", 100))
                self.queue.put(("LOG", f"Sync complete. context.py updated."))
            except Exception as e:
                self.queue.put(("LOG", f"Sync failed: {e}"))

        threading.Thread(target=task, daemon=True).start()

    def _run_job(self, mode):
        strategy = self.strategy_var.get()
        ticker = self.ticker_var.get()
        interval = self.interval_var.get()
        
        if not strategy or not ticker:
            messagebox.showwarning("Input Required", "Please select a strategy and ticker.")
            return

        def task():
            try:
                self.queue.put(("LOG", f"Starting {mode} for {ticker}..."))
                self.queue.put(("PROGRESS", 10))
                
                payload = {
                    "strategy": strategy,
                    "assets": [ticker],
                    "interval": interval,
                    "mode": mode,
                    "timeframe": {"start": None, "end": None}
                }
                
                # Mock high volume output redirection if needed, 
                # but ApplicationController already prints to stdout.
                # We could capture stdout, but for now we rely on explicit log calls.
                results = self.controller.execute_job(payload)
                
                self.queue.put(("LOG", f"{mode} completed."))
                self.queue.put(("LOG", f"Results: {json.dumps(results, indent=2) if results else 'Success'}"))
                self.queue.put(("PROGRESS", 100))
            except Exception as e:
                self.queue.put(("LOG", f"Error during {mode}: {e}"))
                self.queue.put(("PROGRESS", 0))

        threading.Thread(target=task, daemon=True).start()

    def _bundle_artifact(self):
        strategy = self.strategy_var.get()
        if not strategy: return
        
        def task():
            try:
                self.queue.put(("LOG", f"Bundling {strategy}..."))
                strat_path = os.path.join(self.strategies_dir, strategy)
                export_path = "exports"
                bundle_file = Bundler.export(strat_path, export_path)
                self.queue.put(("LOG", f"Artifact created: {bundle_file}"))
            except Exception as e:
                self.queue.put(("LOG", f"Bundling failed: {e}"))

        threading.Thread(target=task, daemon=True).start()

    def _create_new_strategy_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Create New Strategy")
        popup.geometry("300x150")
        
        ttk.Label(popup, text="Strategy Name:").pack(pady=5)
        name_var = tk.StringVar()
        ttk.Entry(popup, textvariable=name_var).pack(pady=5)
        
        def create():
            name = name_var.get().strip().lower().replace(" ", "_")
            if not name: return
            
            strat_path = os.path.join(self.strategies_dir, name)
            if os.path.exists(strat_path):
                messagebox.showerror("Error", "Strategy already exists.")
                return
            
            os.makedirs(strat_path)
            
            # Boilerplate manifest.json
            manifest = {
                "features": [],
                "hyperparameters": {},
                "parameter_bounds": {}
            }
            with open(os.path.join(strat_path, "manifest.json"), 'w') as f:
                json.dump(manifest, f, indent=4)
            
            # Boilerplate model.py
            model_content = """import pandas as pd
import numpy as np
import context as ctx
from src.controller import SignalModel

class MyStrategy(SignalModel):
    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        # Boilerplate logic
        signals = pd.Series(0.0, index=df.index)
        return signals
"""
            with open(os.path.join(strat_path, "model.py"), 'w') as f:
                f.write(model_content)
            
            # Initial context.py
            with open(os.path.join(strat_path, "context.py"), 'w') as f:
                f.write("# AUTO-GENERATED\n")
            
            self._refresh_strategies()
            self.strategy_var.set(name)
            self._on_strategy_selected()
            
            self._log(f"Created new strategy: {name}")
            popup.destroy()

        ttk.Button(popup, text="Create", command=create).pack(pady=10)

    def _listen_to_queue(self):
        try:
            while True:
                msg_type, msg_val = self.queue.get_nowait()
                if msg_type == "LOG":
                    self._log(msg_val)
                elif msg_type == "PROGRESS":
                    self.progress_var.set(msg_val)
        except queue.Empty:
            pass
        self.root.after(100, self._listen_to_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = GuiLauncher(root)
    root.mainloop()
