#!/usr/bin/env bash
# Stop hook: refuse to let the agent finish if its binary fails any
# test that the gold binary passes.
#
# This script runs inside the agent's container (the agent-server invokes
# `Stop` hooks via subprocess on its own side). The cleanroom task image
# stashes the gold binary at $PB_STASHED_GOLD_PATH (default
# /opt/programbench-stashed-executable-do-not-modify) at compile time.
#
# Behaviour:
#   exit 0  -> allow the agent to stop
#   exit 1  -> block the stop, print feedback to stderr (the SDK injects it
#              as an environment MessageEvent and resumes the conversation)
#
# Inputs:
#   stdin              JSON HookEvent (we don't actually need it, but the
#                      SDK pipes it in)
#
#   PB_STASHED_GOLD_PATH         (optional) location of the gold binary
#   PB_AGENT_BINARY_PATH         (optional) location of the agent's binary
#   PB_STOP_HOOK_MAX_RETRIES     (optional) cap re-entries (default: 3)
#   PB_STOP_HOOK_RUNS_DIR        (optional) where to keep state across calls
#                                (default: /workspace/.programbench-stop-hook)
#   PB_STOP_HOOK_TEST_TIMEOUT    (optional) per-pytest-run timeout seconds
#                                (default: 300)
#
# We deliberately do *not* shell out into another container; the agent
# container already has the build tooling and pytest installed by the
# upstream programbench cleanroom image.

set -uo pipefail

GOLD="${PB_STASHED_GOLD_PATH:-/opt/programbench-stashed-executable-do-not-modify}"
AGENT="${PB_AGENT_BINARY_PATH:-./executable}"
RUNS_DIR="${PB_STOP_HOOK_RUNS_DIR:-/workspace/.programbench-stop-hook}"
MAX_RETRIES="${PB_STOP_HOOK_MAX_RETRIES:-3}"
TEST_TIMEOUT="${PB_STOP_HOOK_TEST_TIMEOUT:-300}"

cd "$(dirname "$AGENT")" 2>/dev/null || cd /workspace
mkdir -p "$RUNS_DIR"

# Drain stdin so the SDK doesn't see a SIGPIPE.
cat >/dev/null || true

# --- Retry cap -------------------------------------------------------------
RETRY_FILE="$RUNS_DIR/count"
COUNT=$(cat "$RETRY_FILE" 2>/dev/null || echo 0)
COUNT=$((COUNT + 1))
echo "$COUNT" > "$RETRY_FILE"
if [ "$COUNT" -gt "$MAX_RETRIES" ]; then
    echo "[stop-hook] reached max retries ($MAX_RETRIES); allowing stop" >&2
    exit 0
fi

# --- Quick sanity: agent must have a binary -------------------------------
if [ ! -f "$AGENT" ]; then
    echo "[stop-hook] no agent binary at $AGENT — build your solution before finishing" >&2
    exit 1
fi
if [ ! -x "$AGENT" ]; then
    chmod +x "$AGENT" 2>/dev/null || true
fi

# --- If gold binary isn't available, we can't compare → allow stop --------
if [ ! -f "$GOLD" ]; then
    echo "[stop-hook] gold binary missing at $GOLD; cannot compare; allowing stop" >&2
    exit 0
fi

# --- Cheap path: byte-identical → allow stop ------------------------------
AHASH=$(sha256sum "$AGENT" 2>/dev/null | awk '{print $1}')
GHASH=$(sha256sum "$GOLD"  2>/dev/null | awk '{print $1}')
if [ -n "$AHASH" ] && [ "$AHASH" = "$GHASH" ]; then
    echo "[stop-hook] binary is byte-identical to gold (sha256=$AHASH); allowing stop" >&2
    exit 0
fi

# --- Need an eval/run.sh to compare meaningfully --------------------------
RUN_SH=eval/run.sh
if [ ! -f "$RUN_SH" ]; then
    echo "[stop-hook] no $RUN_SH to run tests against; allowing stop" >&2
    exit 0
fi
chmod +x "$RUN_SH" 2>/dev/null || true

run_branch () {
    # Runs the test suite once and copies the JUnit XML to $1.xml
    # Returns 0 on completion regardless of test outcomes; the caller
    # interprets the XML.
    local out=$1
    rm -f eval/results.xml results.xml
    timeout "$TEST_TIMEOUT" bash "$RUN_SH" > "$out" 2>&1
    local rc=$?
    if [ -f eval/results.xml ]; then
        cp eval/results.xml "$out.xml"
    elif [ -f results.xml ]; then
        cp results.xml "$out.xml"
    fi
    return "$rc"
}

cp "$AGENT" "$RUNS_DIR/agent.bin"
cp "$GOLD"  "$RUNS_DIR/gold.bin"

# Run gold first.
cp -f "$RUNS_DIR/gold.bin" "$AGENT"
chmod +x "$AGENT"
run_branch "$RUNS_DIR/gold.log"

# Run agent.
cp -f "$RUNS_DIR/agent.bin" "$AGENT"
chmod +x "$AGENT"
run_branch "$RUNS_DIR/agent.log"

# --- Compare per-test pass/fail in JUnit XML ------------------------------
GOLD_XML="$RUNS_DIR/gold.log.xml"
AGENT_XML="$RUNS_DIR/agent.log.xml"
if [ ! -f "$GOLD_XML" ] || [ ! -f "$AGENT_XML" ]; then
    echo "[stop-hook] missing JUnit XML (gold=$([ -f "$GOLD_XML" ] && echo yes || echo no), agent=$([ -f "$AGENT_XML" ] && echo yes || echo no)); allowing stop" >&2
    exit 0
fi

# Extract a "tests that pass on gold but not on agent" list using python3.
mapfile -t REPORT < <(python3 - "$GOLD_XML" "$AGENT_XML" <<'PY' || true
import sys
import xml.etree.ElementTree as ET


def read(p):
    out = {}
    try:
        for tc in ET.parse(p).getroot().iter("testcase"):
            name = (tc.get("classname") or "") + "." + (tc.get("name") or "")
            failed = (tc.find("failure") is not None
                      or tc.find("error") is not None)
            out[name] = "fail" if failed else "pass"
    except Exception as exc:  # malformed XML or empty file
        print(f"PARSE_ERROR\t{exc}")
        return None
    return out


g = read(sys.argv[1])
a = read(sys.argv[2])
if g is None or a is None:
    sys.exit(0)
mismatched = sorted(t for t, v in g.items() if v == "pass" and a.get(t) != "pass")
print(f"GOLD_PASS\t{sum(1 for v in g.values() if v == 'pass')}")
print(f"AGENT_PASS\t{sum(1 for v in a.values() if v == 'pass')}")
print(f"MISMATCH\t{len(mismatched)}")
for t in mismatched[:5]:
    print(f"NAME\t{t}")
PY
)

MISMATCH_COUNT=0
EXAMPLES=()
GOLD_PASS=0
AGENT_PASS=0
for line in "${REPORT[@]}"; do
    case "$line" in
        MISMATCH$'\t'*)   MISMATCH_COUNT="${line#*$'\t'}" ;;
        GOLD_PASS$'\t'*)  GOLD_PASS="${line#*$'\t'}" ;;
        AGENT_PASS$'\t'*) AGENT_PASS="${line#*$'\t'}" ;;
        NAME$'\t'*)       EXAMPLES+=("${line#*$'\t'}") ;;
        PARSE_ERROR$'\t'*)
            echo "[stop-hook] could not parse JUnit XML: ${line#*$'\t'}; allowing stop" >&2
            exit 0
            ;;
    esac
done

if [ "$MISMATCH_COUNT" -eq 0 ]; then
    echo "[stop-hook] all gold-passing tests also pass on your binary (gold=$GOLD_PASS agent=$AGENT_PASS); allowing stop" >&2
    exit 0
fi

JOINED=$(IFS=", "; echo "${EXAMPLES[*]:-(none collected)}")
{
    echo "[stop-hook] $MISMATCH_COUNT test(s) pass against the gold binary but fail against your binary."
    echo "First mismatches: $JOINED"
    echo "Keep iterating: rebuild ./executable so it matches gold's behaviour, then signal completion again."
    echo "(retry $COUNT/$MAX_RETRIES)"
} >&2
exit 1
