"""Helpers for preferring bundled GStreamer libraries on macOS."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _prepend_env_path(name: str, values: list[Path]) -> None:
    existing = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    new_values = [str(value) for value in values if value.exists()]
    merged = []
    for item in [*new_values, *existing]:
        if item not in merged:
            merged.append(item)
    if merged:
        os.environ[name] = os.pathsep.join(merged)


def configure_gstreamer_bundle_env() -> None:
    """Prefer pip's bundled GStreamer libraries over older conda libraries."""
    spec = importlib.util.find_spec("gstreamer_libs")
    if spec is None or spec.origin is None:
        return

    site_packages = Path(spec.origin).resolve().parent.parent
    lib_dir = site_packages / "gstreamer_libs" / "lib"
    python_lib_dir = Path(os.__file__).resolve().parents[1]
    plugin_dirs = [
        site_packages / package / "lib" / "gstreamer-1.0"
        for package in (
            "gstreamer_libs",
            "gstreamer_plugins",
            "gstreamer_plugins_libs",
            "gstreamer_plugins_gpl",
            "gstreamer_plugins_restricted",
            "gstreamer_plugins_gpl_restricted",
            "gstreamer_gtk",
            "gstreamer_python",
        )
    ]
    typelib_dirs = [
        site_packages / "gstreamer_libs" / "lib" / "girepository-1.0",
        site_packages / "gstreamer_python" / "lib" / "girepository-1.0",
        site_packages / "gstreamer_gtk" / "lib" / "girepository-1.0",
    ]

    _prepend_env_path("DYLD_LIBRARY_PATH", [lib_dir])
    _prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", [python_lib_dir])
    _prepend_env_path("GST_PLUGIN_SYSTEM_PATH_1_0", plugin_dirs)
    _prepend_env_path("GI_TYPELIB_PATH", typelib_dirs)
