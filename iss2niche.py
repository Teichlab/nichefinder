"""Compatibility layer for the old package name."""

import warnings

warnings.warn(
    "The package 'iss2niche' has been renamed to 'nichefinder'. "
    "Please update your imports to use 'import nichefinder' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from nichefinder import *
