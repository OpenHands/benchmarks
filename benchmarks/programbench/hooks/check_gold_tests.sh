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
#   exit 2  -> block the stop, print feedback to stderr (the SDK injects it
#              as an environment MessageEvent and resumes the conversation)
#
# ⚠ The SDK's Stop-hook contract treats *only* ``exit 2`` as a block
# (see ``openhands/sdk/hooks/executor.py`` -- ``blocked = (rc == 2)``).
# Any other non-zero exit is logged as a hook error but does NOT keep
# the agent running.  The block paths in this script therefore exit 2,
# never exit 1 -- using exit 1 here would silently ignore the rejection
# and let the agent ship a broken submission.  Tests pin this.
#
# Inputs:
#   stdin              NOT read. The SDK wraps this script via
#                      ``bash -s <<'EOF' ... EOF`` (see
#                      ``run_infer.py::_hook_definition_from_script``),
#                      which makes the heredoc itself bash's stdin —
#                      i.e. the rest of THIS script.  We close stdin
#                      with ``exec </dev/null`` below so no descendant
#                      accidentally reads from (and prematurely
#                      consumes) the script source.
#
#   PB_STASHED_GOLD_PATH         (optional) location of the gold binary
#   PB_AGENT_BINARY_PATH         (optional) location of the agent's binary
#   PB_STOP_HOOK_MAX_RETRIES     (optional) cap re-entries (default: 20).
#                                The agent's per-conversation iteration
#                                budget already bounds runaway loops, so
#                                this only guards against a hook whose
#                                feedback the agent ignores.
#   PB_STOP_HOOK_RUNS_DIR        (optional) where to keep state across calls.
#                                MUST be outside /workspace — the orchestrator
#                                tars /workspace immediately after this hook
#                                returns, so any churn here races with tar
#                                and triggers ``tar: .: file changed as we
#                                read it``. Default: /tmp/programbench-stop-hook
#   PB_STOP_HOOK_TEST_TIMEOUT    (optional) per-pytest-run timeout seconds
#                                (default: 300)
#
# We deliberately do *not* shell out into another container; the agent
# container already has the build tooling and pytest installed by the
# upstream programbench cleanroom image.
#
# ⚠ Workspace isolation contract (the reason this script is more involved
# than it looks): the orchestrator creates ``submission.tar.gz`` from
# /workspace immediately after the hook returns. Any files we add/delete/
# rename in /workspace.  during the hook race with that tar invocation and
# crash the run with ``file changed as we read it``. Therefore we:
#   1. Copy /workspace to a scratch dir under /tmp,
#   2. Do all binary-swapping and pytest-running inside the scratch dir,
#   3. Tear down the scratch dir on exit.
# /workspace is read-only as far as this hook is concerned.

set -uo pipefail

GOLD="${PB_STASHED_GOLD_PATH:-/opt/programbench-stashed-executable-do-not-modify}"
AGENT="${PB_AGENT_BINARY_PATH:-./executable}"
# RUNS_DIR holds cross-invocation state (the retry counter). Placing it
# under /tmp instead of /workspace guarantees it can't pollute the
# workspace tarball even if a hook invocation is killed mid-write.
RUNS_DIR="${PB_STOP_HOOK_RUNS_DIR:-/tmp/programbench-stop-hook}"
MAX_RETRIES="${PB_STOP_HOOK_MAX_RETRIES:-20}"
TEST_TIMEOUT="${PB_STOP_HOOK_TEST_TIMEOUT:-300}"

# Resolve AGENT to an absolute path before we cd anywhere — every
# subsequent reference uses $WORKSPACE / $SCRATCH to dereference.
WORKSPACE="$(cd "$(dirname "$AGENT")" 2>/dev/null && pwd)"
[ -z "$WORKSPACE" ] && WORKSPACE=/workspace
AGENT_NAME="$(basename "$AGENT")"

mkdir -p "$RUNS_DIR"

# IMPORTANT: do NOT read or redirect stdin from this script.
#
# The SDK wraps this hook via ``bash -s <<'PROGRAMBENCH_HOOK_EOF' ...
# PROGRAMBENCH_HOOK_EOF`` (see ``run_infer.py::
# _hook_definition_from_script``).  Under ``bash -s`` bash reads the
# script body itself from stdin (the heredoc).  Anything that consumes
# stdin here — ``cat >/dev/null``, ``exec </dev/null``, ``read line``
# — consumes the rest of THIS script's source, after which bash hits
# EOF on the next read and silently exits 0 BEFORE any of the gold/
# agent comparison below ever runs, turning the hook into a no-op
# that green-lights every broken submission.
#
# We don't need the JSON HookEvent the SDK pipes to the parent shell
# anyway (it goes to /bin/sh, not to bash), so just leave stdin alone.
# A regression test pins this:
# ``tests/test_programbench.py::
# test_hooks_actually_run_under_bash_dash_s_heredoc``.

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
# (All subsequent file reads use the original $WORKSPACE; we never modify
# anything there.)
if [ ! -f "$WORKSPACE/$AGENT_NAME" ]; then
    echo "[stop-hook] no agent binary at $WORKSPACE/$AGENT_NAME — build your solution before finishing" >&2
    exit 2
fi

# --- If gold binary isn't available, we can't compare → allow stop --------
if [ ! -f "$GOLD" ]; then
    echo "[stop-hook] gold binary missing at $GOLD; cannot compare; allowing stop" >&2
    exit 0
fi

# --- Cheap path: byte-identical → allow stop ------------------------------
AHASH=$(sha256sum "$WORKSPACE/$AGENT_NAME" 2>/dev/null | awk '{print $1}')
GHASH=$(sha256sum "$GOLD"  2>/dev/null | awk '{print $1}')
if [ -n "$AHASH" ] && [ "$AHASH" = "$GHASH" ]; then
    echo "[stop-hook] binary is byte-identical to gold (sha256=$AHASH); allowing stop" >&2
    exit 0
fi

# --- Need an eval/run.sh to compare meaningfully --------------------------
if [ ! -f "$WORKSPACE/eval/run.sh" ]; then
    echo "[stop-hook] no eval/run.sh to run tests against; allowing stop" >&2
    exit 0
fi

# --- Workspace-isolated test runs ----------------------------------------
# Materialise a copy of the workspace under /tmp and do ALL test work
# there. /workspace stays bit-for-bit identical from this point on, so
# the orchestrator's submission tarball can't race with us.
SCRATCH=$(mktemp -d /tmp/pb-stop-hook-scratch.XXXXXX) || {
    echo "[stop-hook] could not allocate scratch dir; allowing stop" >&2
    exit 0
}
# Always tear down — even on early exits — so /tmp stays clean.
trap 'rm -rf "$SCRATCH" 2>/dev/null || true' EXIT

# `cp -a` preserves modes/symlinks/timestamps, which matters because
# eval/run.sh often hardcodes ./executable and executable bits.
if ! cp -a "$WORKSPACE/." "$SCRATCH/" 2>"$RUNS_DIR/cp.err"; then
    echo "[stop-hook] could not stage workspace into scratch dir: $(cat "$RUNS_DIR/cp.err" 2>/dev/null | head -3); allowing stop" >&2
    exit 0
fi
chmod +x "$SCRATCH/$AGENT_NAME" "$SCRATCH/eval/run.sh" 2>/dev/null || true
cd "$SCRATCH"

run_branch () {
    # Runs the test suite once in $SCRATCH and copies the JUnit XML to $1.xml
    # Returns 0 on completion regardless of test outcomes; the caller
    # interprets the XML.
    local out=$1
    rm -f eval/results.xml results.xml
    timeout "$TEST_TIMEOUT" bash eval/run.sh > "$out" 2>&1
    local rc=$?
    if [ -f eval/results.xml ]; then
        cp eval/results.xml "$out.xml"
    elif [ -f results.xml ]; then
        cp results.xml "$out.xml"
    fi
    return "$rc"
}

# Stash the agent's binary so we can swap freely inside the scratch.
cp "$SCRATCH/$AGENT_NAME" "$RUNS_DIR/agent.bin"
cp "$GOLD"               "$RUNS_DIR/gold.bin"

# Gold branch.
cp -f "$RUNS_DIR/gold.bin" "$SCRATCH/$AGENT_NAME"
chmod +x "$SCRATCH/$AGENT_NAME"
run_branch "$RUNS_DIR/gold.log"

# Agent branch.
cp -f "$RUNS_DIR/agent.bin" "$SCRATCH/$AGENT_NAME"
chmod +x "$SCRATCH/$AGENT_NAME"
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
exit 2
