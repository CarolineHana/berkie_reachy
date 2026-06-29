"""Runtime patches for local Reachy Mini development.

This file is imported automatically by Python when it is on ``sys.path``.
We use it to redirect Reachy Mini's local app metadata directory away from
the system Python installation prefix, which is not writable in this
environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def _fix_ssl_certs() -> None:
    # Python 3.11 from python.org on macOS ships without a cert.pem at the
    # expected openssl path; certifi provides the bundle we need.
    if not os.environ.get("SSL_CERT_FILE"):
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass


_fix_ssl_certs()


def _patch_reachy_mini_app_metadata_dir() -> None:
    runtime_root = Path("/private/tmp/berkie_reachy_runtime")
    runtime_root.mkdir(parents=True, exist_ok=True)

    try:
        from reachy_mini.apps.sources import local_common_venv
    except Exception:
        return

    def _get_venv_parent_dir() -> Path:
        return runtime_root

    local_common_venv._get_venv_parent_dir = _get_venv_parent_dir  # type: ignore[attr-defined]


_patch_reachy_mini_app_metadata_dir()
