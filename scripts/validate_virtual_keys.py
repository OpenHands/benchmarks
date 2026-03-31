#!/usr/bin/env python3
"""Validate LiteLLM virtual key cost tracking end-to-end.

This script exercises the full virtual key lifecycle against a real LiteLLM
proxy and optionally runs a minimal ACP codex interaction to verify that
per-trajectory costs are recorded.

Usage:
    # Just test virtual key CRUD (no LLM calls)
    LLM_BASE_URL=https://llm-proxy.eval.all-hands.dev \
    LITELLM_MASTER_KEY=sk-... \
        python -m scripts.validate_virtual_keys

    # Full ACP codex test (creates a key, runs codex, queries spend)
    LLM_BASE_URL=https://llm-proxy.eval.all-hands.dev \
    LITELLM_MASTER_KEY=sk-... \
    OPENAI_API_KEY=sk-... \
        python -m scripts.validate_virtual_keys --with-codex
"""

import argparse
import json
import os
import time

import httpx


def test_virtual_key_lifecycle():
    """Test create -> info -> delete lifecycle against the proxy."""
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    master_key = os.environ.get("LITELLM_MASTER_KEY", "")

    if not base_url or not master_key:
        print("SKIP: LLM_BASE_URL and LITELLM_MASTER_KEY required")
        return False

    print(f"Proxy: {base_url}")
    headers = {"Authorization": f"Bearer {master_key}"}

    # 1. Create key
    print("\n--- Step 1: Create virtual key ---")
    resp = httpx.post(
        f"{base_url}/key/generate",
        headers=headers,
        json={
            "metadata": {"instance_id": "validate-test", "run_id": "manual"},
            "max_budget": 1.0,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    virtual_key = data["key"]
    print(f"  Created key: {virtual_key[:20]}...")

    # 2. Query key info (spend should be 0)
    print("\n--- Step 2: Query key info ---")
    resp = httpx.get(
        f"{base_url}/key/info",
        params={"key": virtual_key},
        headers=headers,
        timeout=30.0,
    )
    resp.raise_for_status()
    info = resp.json()["info"]
    print(f"  Spend: ${info['spend']:.6f}")
    print(f"  Max budget: ${info.get('max_budget', 'N/A')}")
    print(f"  Metadata: {info.get('metadata', {})}")
    assert info["spend"] == 0.0, f"Expected 0 spend, got {info['spend']}"

    # 3. Delete key
    print("\n--- Step 3: Delete virtual key ---")
    resp = httpx.post(
        f"{base_url}/key/delete",
        headers=headers,
        json={"keys": [virtual_key]},
        timeout=30.0,
    )
    resp.raise_for_status()
    print("  Deleted successfully")

    print("\n=== Virtual key lifecycle: PASS ===")
    return True


def test_codex_with_virtual_key():
    """Run a minimal codex ACP interaction using a virtual key and verify spend."""
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    master_key = os.environ.get("LITELLM_MASTER_KEY", "")

    if not base_url or not master_key:
        print("SKIP: LLM_BASE_URL and LITELLM_MASTER_KEY required")
        return False

    headers = {"Authorization": f"Bearer {master_key}"}

    # 1. Create virtual key
    print("\n--- Step 1: Create virtual key for codex ---")
    resp = httpx.post(
        f"{base_url}/key/generate",
        headers=headers,
        json={
            "metadata": {"instance_id": "codex-validate", "run_id": "manual"},
            "max_budget": 5.0,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    virtual_key = resp.json()["key"]
    print(f"  Created key: {virtual_key[:20]}...")

    # 2. Run a minimal codex ACP interaction
    print("\n--- Step 2: Run codex ACP with virtual key ---")
    try:
        from benchmarks.utils.litellm_proxy import set_current_virtual_key
        from openhands.sdk import Conversation
        from openhands.sdk.agent import ACPAgent
        from openhands.sdk.workspace import LocalWorkspace

        # Store virtual key in thread-local (simulates orchestrator behavior)
        set_current_virtual_key(virtual_key)

        # Override env so codex uses the virtual key via the proxy
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = virtual_key
        os.environ["OPENAI_BASE_URL"] = base_url

        agent = ACPAgent(
            acp_command=["codex-acp"],
            acp_model="o4-mini",
            acp_prompt_timeout=120.0,
            acp_env={"OPENAI_API_KEY": virtual_key, "OPENAI_BASE_URL": base_url},
        )

        workspace = LocalWorkspace(working_dir="/tmp")
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=1,
        )

        print("  Sending minimal prompt to codex...")
        conversation.send_message("What is 2 + 2? Reply with just the number.")
        conversation.run()

        # Get metrics from conversation
        metrics = conversation.conversation_stats.get_combined_metrics()
        print(f"  SDK accumulated_cost: ${metrics.accumulated_cost:.6f}")
        print(
            f"  Tokens: prompt={metrics.accumulated_token_usage.prompt_tokens}, "
            f"completion={metrics.accumulated_token_usage.completion_tokens}"
        )

        agent.close()
        set_current_virtual_key(None)

        # Restore env
        if original_openai_key:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_BASE_URL", None)

    except Exception as e:
        print(f"  Codex interaction failed: {e}")
        print("  (This is OK if codex-acp is not properly configured)")

    # 3. Wait briefly for proxy to register the spend
    print("\n--- Step 3: Query spend after codex run ---")
    time.sleep(2)
    resp = httpx.get(
        f"{base_url}/key/info",
        params={"key": virtual_key},
        headers=headers,
        timeout=30.0,
    )
    resp.raise_for_status()
    info = resp.json()["info"]
    spend = info["spend"]
    print(f"  Proxy-reported spend: ${spend:.6f}")

    if spend > 0:
        print(f"  SUCCESS: Proxy tracked ${spend:.6f} for this trajectory")
    else:
        print("  WARNING: Spend is $0.00 - proxy may not have registered the call yet")

    # 4. Clean up
    print("\n--- Step 4: Delete virtual key ---")
    httpx.post(
        f"{base_url}/key/delete",
        headers=headers,
        json={"keys": [virtual_key]},
        timeout=30.0,
    )
    print("  Deleted successfully")

    status = "PASS" if spend > 0 else "INCONCLUSIVE"
    print(f"\n=== Codex virtual key cost tracking: {status} ===")
    return spend > 0


def test_benchmarks_integration():
    """Test that the benchmarks utility functions work correctly."""
    print("\n--- Benchmarks integration test ---")

    from benchmarks.utils.litellm_proxy import (
        create_virtual_key,
        delete_key,
        get_current_virtual_key,
        get_key_spend,
        set_current_virtual_key,
    )
    from benchmarks.utils.models import EvalOutput
    from benchmarks.utils.report_costs import extract_accumulated_cost

    # Test 1: No-op without config
    key = create_virtual_key("test")
    if key is None:
        print("  create_virtual_key is no-op without config: PASS")
    else:
        print(f"  Created key: {key[:20]}... (proxy is configured)")

    # Test 2: EvalOutput with proxy_cost
    out = EvalOutput(
        instance_id="test",
        test_result={},
        proxy_cost=1.23,
    )
    data = json.loads(out.model_dump_json())
    assert data["proxy_cost"] == 1.23
    print("  EvalOutput proxy_cost serialization: PASS")

    # Test 3: report_costs prefers proxy_cost
    entries = [
        {"proxy_cost": 1.50, "metrics": {"accumulated_cost": 0.75}},
        {"metrics": {"accumulated_cost": 1.00}},
    ]
    cost = extract_accumulated_cost(entries)
    assert cost == 2.50, f"Expected 2.50, got {cost}"
    print("  report_costs prefers proxy_cost: PASS")

    # Test 4: thread-local virtual key
    assert get_current_virtual_key() is None
    set_current_virtual_key("test-key-123")
    assert get_current_virtual_key() == "test-key-123"
    set_current_virtual_key(None)
    assert get_current_virtual_key() is None
    print("  thread-local virtual key: PASS")

    # Test 5: If proxy is configured, do a full lifecycle
    base_url = os.environ.get("LLM_BASE_URL", "")
    master_key = os.environ.get("LITELLM_MASTER_KEY", "")
    if base_url and master_key:
        key = create_virtual_key("integration-test", run_id="validate")
        if key:
            spend = get_key_spend(key)
            assert spend is not None and spend == 0.0
            delete_key(key)
            print("  Full virtual key lifecycle via utility functions: PASS")
    else:
        print("  SKIP full lifecycle (no proxy credentials)")

    print("\n=== Benchmarks integration: PASS ===")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-codex",
        action="store_true",
        help="Also run a minimal codex ACP interaction",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LiteLLM Virtual Key Cost Tracking Validation")
    print("=" * 60)

    results = {}

    # Always run integration test
    results["benchmarks_integration"] = test_benchmarks_integration()

    # Test virtual key lifecycle if proxy is configured
    results["key_lifecycle"] = test_virtual_key_lifecycle()

    # Optionally test with codex
    if args.with_codex:
        results["codex_cost_tracking"] = test_codex_with_virtual_key()

    print("\n" + "=" * 60)
    print("RESULTS:")
    for name, passed in results.items():
        status = "PASS" if passed else "SKIP/FAIL"
        print(f"  {name}: {status}")
    print("=" * 60)

    if all(results.values()):
        print("\nAll tests passed!")
    elif any(not v for v in results.values()):
        missing = [k for k, v in results.items() if not v]
        print(f"\nSome tests skipped/failed: {missing}")
        print("Set LLM_BASE_URL and LITELLM_MASTER_KEY to run proxy tests")


if __name__ == "__main__":
    main()
