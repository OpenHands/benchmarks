#!/usr/bin/env python3
"""
Cleanup script to stop all running runtimes for your API key.
This is useful when runtimes are orphaned after local script termination.
"""

import os
from typing import Any, Dict, List

import httpx


# Runtime API configuration
RUNTIME_API_URL = os.getenv("RUNTIME_API_URL", "https://runtime.eval.all-hands.dev")
RUNTIME_API_KEY = os.getenv("RUNTIME_API_KEY")

if not RUNTIME_API_KEY:
    print("ERROR: RUNTIME_API_KEY environment variable not set")
    print("Please set it with: export RUNTIME_API_KEY=your_api_key")
    exit(1)


def list_runtimes() -> List[Dict[str, Any]]:
    """List all runtimes for this API key."""
    headers = {"X-API-Key": RUNTIME_API_KEY}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{RUNTIME_API_URL}/list", headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("runtimes", [])
    except Exception as e:
        print(f"Error listing runtimes: {e}")
        return []


def stop_runtime(runtime_id: str) -> bool:
    """Stop a specific runtime."""
    headers = {"X-API-Key": RUNTIME_API_KEY}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{RUNTIME_API_URL}/stop",
                json={"runtime_id": runtime_id},
                headers=headers,
            )
            response.raise_for_status()
            return True
    except Exception as e:
        print(f"Error stopping runtime {runtime_id}: {e}")
        return False


def main():
    print(f"Fetching runtimes from {RUNTIME_API_URL}...")
    runtimes = list_runtimes()

    if not runtimes:
        print("No runtimes found or error occurred.")
        return

    running_runtimes = [r for r in runtimes if r.get("status") == "running"]

    print(f"\nFound {len(runtimes)} total runtimes, {len(running_runtimes)} running")

    if not running_runtimes:
        print("No running runtimes to clean up.")
        return

    print("\nRunning runtimes:")
    for rt in running_runtimes:
        runtime_id = rt.get("runtime_id", "unknown")
        created = rt.get("created_at", "unknown")
        image = rt.get("image", "unknown")
        print(f"  - {runtime_id}: {image} (created: {created})")

    response = input(f"\nStop all {len(running_runtimes)} running runtimes? [y/N]: ")

    if response.lower() != "y":
        print("Aborted.")
        return

    print("\nStopping runtimes...")
    succeeded = 0
    failed = 0

    for rt in running_runtimes:
        runtime_id = rt.get("runtime_id", "unknown")
        print(f"  Stopping {runtime_id}...", end=" ")

        if stop_runtime(runtime_id):
            print("✓")
            succeeded += 1
        else:
            print("✗")
            failed += 1

    print(f"\nResults: {succeeded} stopped, {failed} failed")


if __name__ == "__main__":
    main()
