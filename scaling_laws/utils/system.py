"""System/runtime info printers."""

from __future__ import annotations

import platform

import tensorflow as tf


def print_system_info():
    """Print system and TensorFlow information."""
    print("=" * 80)
    print("NEURAL NETWORK SCALING LAWS")
    print("=" * 80)
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    try:
        policy = tf.keras.mixed_precision.global_policy()
        print(
            "Keras precision policy: "
            f"{policy.name} (compute={policy.compute_dtype}, variable={policy.variable_dtype})"
        )
    except Exception as exc:
        print(f"Keras precision policy: unavailable ({exc})")
    try:
        tf32_enabled = tf.config.experimental.tensor_float_32_execution_enabled()
        print(f"NVIDIA TF32 enabled: {tf32_enabled}")
    except Exception as exc:
        print(f"NVIDIA TF32 enabled: unavailable ({exc})")
    print("=" * 80)
    print()
