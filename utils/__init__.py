"""Utility package.

Heavy optional dependencies are imported by their concrete modules instead of
package import time, so lightweight CLIs can run without importing numpy/torch.
"""
