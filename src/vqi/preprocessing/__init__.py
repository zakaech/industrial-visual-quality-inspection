"""Image preprocessing shared by training, evaluation and the inference API.

There is exactly one code path from an image (file or bytes) to a model input:
:class:`vqi.preprocessing.pipeline.Preprocessor`. Anything else would be train/serve skew.
"""

from vqi.preprocessing.pipeline import Preprocessor

__all__ = ["Preprocessor"]
