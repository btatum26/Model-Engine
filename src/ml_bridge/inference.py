# Phase 3.5: ML Training Pipeline - Inference
# numpy.tanh normalizer for signal arrays

import numpy as np

def normalize_signals(signals: np.ndarray) -> np.ndarray:
    """Apply numpy.tanh normalizer to signal arrays."""
    return np.tanh(signals)
