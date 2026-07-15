"""Standalone CLI for testing the llm_engine bootstrap without any ReachyMiniApp wiring.

Usage:
    berkie-llm-backend-check --bedrock-api-key ... --bedrock-base-url ...

Reads BEDROCK_API_KEY / BEDROCK_BASE_URL from the environment if the flags
are omitted. Runs one synchronous bootstrap pass, printing progress, and
exits 0 on success or 1 if anything required is missing/failed.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging

from berkie_reachy.llm_engine_bootstrap import run_bootstrap


def main() -> None:
    """Entry point for the `berkie-llm-backend-check` console script."""
    parser = argparse.ArgumentParser(description="Provision and verify the local Berky llm_engine backend.")
    parser.add_argument("--bedrock-api-key", default=os.getenv("BEDROCK_API_KEY", ""))
    parser.add_argument("--bedrock-base-url", default=os.getenv("BEDROCK_BASE_URL", ""))
    parser.add_argument("--instance-path", default=None, help="Directory to write .env config into.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    def on_progress(step: str, message: str) -> None:
        print(f"[{step}] {message}")

    result = run_bootstrap(
        instance_path=args.instance_path,
        bedrock_api_key=args.bedrock_api_key,
        bedrock_base_url=args.bedrock_base_url,
        on_progress=on_progress,
    )

    if result.skipped:
        print("Bootstrap did not complete - see messages above for what's missing.", file=sys.stderr)
        sys.exit(1)

    print(f"Berky is ready. Conversation ID: {result.conversation_id}")
    sys.exit(0)


if __name__ == "__main__":
    main()
