"""Memory-management helpers.

``malloc_trim`` is a Linux-only operation. On macOS/Windows it silently
no-ops because ``libc.so.6`` is unavailable.
"""

from __future__ import annotations

import ctypes
import gc

import matplotlib.pyplot as plt
import tensorflow as tf

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryManager:
    """Utility class for memory management operations."""

    @staticmethod
    def malloc_trim():
        """Force glibc to return memory to OS (Linux only)."""
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    @staticmethod
    def print_memory_usage(label: str = ""):
        """Print current memory usage."""
        if not PSUTIL_AVAILABLE:
            return
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"[MEMORY {label}] RSS: {mem_info.rss / 1024 ** 3:.2f} GB, "
                  f"VMS: {mem_info.vms / 1024 ** 3:.2f} GB")
        except Exception:
            pass

    @staticmethod
    def aggressive_cleanup():
        """Perform aggressive memory cleanup."""
        # Clear TensorFlow session
        tf.keras.backend.clear_session()

        # Reset default graph (TF1 compatibility)
        try:
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass

        # Force Python garbage collection - run multiple times for circular refs
        gc.collect()
        gc.collect()
        gc.collect()

        # Return memory to OS (Linux)
        MemoryManager.malloc_trim()

        # Close all matplotlib figures
        plt.close('all')
