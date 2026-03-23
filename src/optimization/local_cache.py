import ray
import pandas as pd

class LocalCache:
    """
    Phase 3.5: PyArrow Parquet RAM Cache.
    Protects SQLite from concurrency locks during parallel Ray trials.
    Provides Zero-Copy access via Ray's Plasma Store.
    """
    
    def __init__(self):
        self._refs = {}

    def load_to_ram(self, dataset_ref: str, df: pd.DataFrame) -> ray.ObjectRef:
        """
        Caches raw OHLCV data once per asset using ray.put for Plasma Store management.
        """
        ref = ray.put(df)
        self._refs[dataset_ref] = ref
        return ref

    def get_ref(self, dataset_ref: str) -> ray.ObjectRef:
        """
        Provides the reference pointer for workers to consume.
        """
        return self._refs.get(dataset_ref)

    def clear_cache(self, dataset_ref: str) -> None:
        """
        Removes the object from Plasma store and cleans up local dictionary.
        Note: Ray automatically garbage collects objects when no more references exist.
        """
        if dataset_ref in self._refs:
            # We remove the reference from our dictionary. 
            # In larger clusters, we might call ray.internal.free or similar.
            del self._refs[dataset_ref]
