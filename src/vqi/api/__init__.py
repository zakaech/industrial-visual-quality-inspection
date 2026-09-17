"""FastAPI inference service around the frozen model bundle.

The service never re-implements preprocessing, scoring or thresholding: every request goes
through :class:`vqi.inference.Predictor`, the same object that produced the final test metrics.
"""

from vqi.api.app import create_app

__all__ = ["create_app"]
