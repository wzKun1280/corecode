"""Command-line entry point for the local weather-log agent."""

from __future__ import annotations

import argparse
import json
import sys

from weather_agent import ToolError, WeatherLogAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query weather_log.txt with local Ollama")
    parser.add_argument("query", nargs="?", help="Natural-language weather-log query")
    parser.add_argument("--log", default="weather_log.txt", help="Path to the weather log")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the Ollama request (useful for deterministic local tool tests)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = args.query or input("query> ")
    try:
        calls = WeatherLogAgent(log_path=args.log).run_stream(
            query, consult_model=not args.offline
        )
        # Emit each tool call immediately after that tool completes.
        for call in calls:
            print(json.dumps(call, ensure_ascii=False), flush=True)
    except ToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
