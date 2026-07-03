"""System/runtime info printers."""

from __future__ import annotations

import tensorflow as tf


def print_system_info():
    """Print system and TensorFlow information."""
    print("=" * 80)
    print("NEURAL NETWORK SCALING LAWS")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print("=" * 80)
    print()
