"""Install a launchd agent for scheduled StateGraph index refresh on macOS."""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path


def _start_interval_seconds(cadence: str) -> int:
    if cadence == "weekly":
        return 7 * 24 * 60 * 60
    return 24 * 60 * 60


def main() -> int:
    parser = argparse.ArgumentParser(description="Install launchd schedule for StateGraph index refresh.")
    parser.add_argument(
        "--label",
        default="com.naturalist-companion.stategraph.refresh",
        help="launchd label (default: com.naturalist-companion.stategraph.refresh).",
    )
    parser.add_argument(
        "--cadence",
        choices=("daily", "weekly"),
        default="daily",
        help="Refresh cadence (default: daily).",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("deterministic", "realistic"),
        default="realistic",
        help="Refresh source mode (default: realistic).",
    )
    parser.add_argument(
        "--partitions",
        default="all",
        help="Comma-separated partitions or 'all' (default: all).",
    )
    parser.add_argument(
        "--store-root",
        default="out/stategraph_store",
        help="Persistent retrieval store root (default: out/stategraph_store).",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Repository working directory used by launchd command (default: current dir).",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Run launchctl bootstrap after writing plist.",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("This installer is macOS-only (launchd).")
        return 1

    workdir = Path(args.workdir).expanduser().resolve()
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / f"{args.label}.plist"
    log_root = Path(args.store_root).expanduser().resolve() / "launchd_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    command = (
        f"cd {workdir} && "
        "uv run python scripts/stategraph_refresh_index.py "
        f"--runtime-mode {args.runtime_mode} "
        f"--partitions {args.partitions} "
        f"--store-root {Path(args.store_root).expanduser().resolve()} "
        f"--refresh-cadence {args.cadence}"
    )
    payload = {
        "Label": args.label,
        "ProgramArguments": ["/bin/zsh", "-lc", command],
        "RunAtLoad": True,
        "StartInterval": _start_interval_seconds(args.cadence),
        "WorkingDirectory": str(workdir),
        "StandardOutPath": str(log_root / "stdout.log"),
        "StandardErrorPath": str(log_root / "stderr.log"),
    }

    plist_path.write_bytes(plistlib.dumps(payload))
    print("Installed launchd plist")
    print(f"- label: {args.label}")
    print(f"- plist: {plist_path}")
    print(f"- cadence: {args.cadence}")
    print(f"- command: {command}")

    if args.load:
        try:
            subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(plist_path)], check=False)
        except Exception:
            pass
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)], check=True)
        print("- launchctl: bootstrapped")
    else:
        print("- launchctl: not loaded (pass --load to bootstrap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
