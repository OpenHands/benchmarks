#!/usr/bin/env bash
# Stop hook: refuse to let the agent finish unless ProgramBench's build
# contract is satisfied. The grader runs
#
#     chmod +x ./compile.sh && ./compile.sh
#
# from a freshly-extracted submission tar and expects ./compile.sh to
# exit 0 and produce an executable at ./executable. If either is
# missing the grader marks the instance ``compile_failed`` and every
# test branch errors out (``no_expected_test_list``), turning a working
# solution into 0/100. This hook closes that gap by validating the
# contract end-to-end before allowing the agent to stop.
#
# Always-on counterpart to the heavier, opt-in
# ``check_gold_tests.sh``. Keeping the two split lets us:
#   * enforce the cheap, universal contract check on every run, and
#   * opt in to the expensive gold-vs-agent test comparison only when
#     ``--enforce-gold-tests`` is on.
#
# Behaviour:
#   exit 0  -> allow the agent to stop
#   exit 2  -> block the stop, print feedback to stderr (the SDK
#              injects it as an environment MessageEvent and resumes
#              the conversation)
#
# ⚠ The SDK's Stop-hook contract treats *only* ``exit 2`` as a block
# (see ``openhands/sdk/hooks/executor.py`` -- ``blocked = (rc == 2)``).
# Any other non-zero exit is logged as a hook error but does NOT keep
# the agent running.  The block paths in this script therefore exit 2,
# never exit 1 -- using exit 1 here would silently ignore the rejection
# and let the agent ship a broken submission.  Tests pin this.
#
# Inputs:
#   stdin                          NOT read. When the SDK wraps the
#                                  hook in ``bash -s <<EOF ... EOF``,
#                                  bash's stdin IS the heredoc — the
#                                  rest of this script. We close it
#                                  with ``exec </dev/null`` below so
#                                  no descendant accidentally reads
#                                  from (and prematurely consumes) it.
#   PB_WORKSPACE                   (default: /workspace) workspace root
#   PB_COMPILE_HOOK_MAX_RETRIES    (default: 20) cap re-entries; the
#                                  agent's per-conversation iteration
#                                  budget already bounds runaway loops,
#                                  so this only guards against a hook
#                                  whose feedback the agent ignores.
#   PB_COMPILE_HOOK_RUNS_DIR       (default: /tmp/programbench-compile-hook)
#                                  MUST be outside $PB_WORKSPACE — the
#                                  orchestrator tars $PB_WORKSPACE
#                                  immediately after this hook returns,
#                                  so any churn here races with tar and
#                                  trips ``tar: .: file changed as we
#                                  read it``.
#   PB_COMPILE_HOOK_TIMEOUT        (default: 1800) compile.sh timeout secs
#
# ⚠ Workspace isolation contract: see the longer note in
# ``check_gold_tests.sh``. We copy $PB_WORKSPACE to a scratch dir under
# /tmp before running compile.sh, so the workspace stays bit-for-bit
# identical and the orchestrator's submission tarball can't race with us.

set -uo pipefail

WORKSPACE="${PB_WORKSPACE:-/workspace}"
# State (retry counter, build log) lives outside the workspace so it
# can't pollute the submission tarball.
RUNS_DIR="${PB_COMPILE_HOOK_RUNS_DIR:-/tmp/programbench-compile-hook}"
MAX_RETRIES="${PB_COMPILE_HOOK_MAX_RETRIES:-20}"
TIMEOUT="${PB_COMPILE_HOOK_TIMEOUT:-1800}"

if [ ! -d "$WORKSPACE" ]; then
    echo "[compile-contract] $WORKSPACE does not exist; allowing stop" >&2
    exit 0
fi
mkdir -p "$RUNS_DIR"

# IMPORTANT: do NOT read or redirect stdin from this script.
#
# The SDK wraps this hook via ``bash -s <<'PROGRAMBENCH_HOOK_EOF' ...
# PROGRAMBENCH_HOOK_EOF`` (see ``run_infer.py::
# _hook_definition_from_script``).  Under ``bash -s`` bash reads the
# script body itself from stdin (the heredoc).  Anything that consumes
# stdin here — ``cat >/dev/null``, ``exec </dev/null``, ``read line``
# — consumes the rest of THIS script's source, after which bash hits
# EOF on the next read and silently exits 0 BEFORE any of the
# contract checks below ever runs, turning the hook into a no-op
# that green-lights every broken submission.
#
# We don't need the JSON HookEvent the SDK pipes to the parent shell
# anyway (it goes to /bin/sh, not to bash), so just leave stdin alone.
# A regression test pins this:
# ``tests/test_programbench.py::
# test_hooks_actually_run_under_bash_dash_s_heredoc``.

# --- Retry cap ------------------------------------------------------------
RETRY_FILE="$RUNS_DIR/count"
COUNT=$(cat "$RETRY_FILE" 2>/dev/null || echo 0)
COUNT=$((COUNT + 1))
echo "$COUNT" > "$RETRY_FILE"
if [ "$COUNT" -gt "$MAX_RETRIES" ]; then
    echo "[compile-contract] reached max retries ($MAX_RETRIES); allowing stop" >&2
    exit 0
fi

# --- 1. compile.sh must exist (read-only check on $WORKSPACE) -----------
if [ ! -f "$WORKSPACE/compile.sh" ]; then
    {
        echo "[compile-contract] $WORKSPACE/compile.sh is missing."
        echo
        echo "ProgramBench's eval harness builds your submission with:"
        echo
        echo "    chmod +x ./compile.sh && ./compile.sh"
        echo
        echo "from a fresh extraction of your tarball, and requires that"
        echo "the script produce an executable at ./executable. Without"
        echo "compile.sh every test branch errors out (compile_failed)"
        echo "and the instance scores 0/100 — even when the underlying"
        echo "code is correct."
        echo
        echo "Fix: write a Bash script at ./compile.sh that builds your"
        echo "project end-to-end and copies the resulting binary to"
        echo "./executable. Examples:"
        echo
        echo "  # Rust"
        echo "  cargo build --release && cp target/release/<binname> ./executable"
        echo
        echo "  # C/Make"
        echo "  make && cp <binname> ./executable"
        echo
        echo "Then signal completion again."
        echo "(retry $COUNT/$MAX_RETRIES)"
    } >&2
    exit 2
fi

# --- 2. compile.sh must build cleanly and produce ./executable ----------
# Materialise a scratch copy of $WORKSPACE under /tmp and run compile.sh
# there. $WORKSPACE stays bit-for-bit identical, so the orchestrator's
# subsequent submission tarball cannot race with our build artifacts
# (target/, build/, *.o, ./executable, etc.). See the workspace
# isolation contract note at the top of the file.
SCRATCH=$(mktemp -d /tmp/pb-compile-hook-scratch.XXXXXX) || {
    echo "[compile-contract] could not allocate scratch dir; allowing stop" >&2
    exit 0
}
trap 'rm -rf "$SCRATCH" 2>/dev/null || true' EXIT

# `cp -a` preserves modes/symlinks/timestamps. We copy the agent's
# ./compile.sh (and source tree) — *not* any pre-built artefacts the
# agent might have produced manually — by wiping ./executable in the
# scratch only. $WORKSPACE/executable is left alone.
if ! cp -a "$WORKSPACE/." "$SCRATCH/" 2>"$RUNS_DIR/cp.err"; then
    echo "[compile-contract] could not stage workspace into scratch dir: $(cat "$RUNS_DIR/cp.err" 2>/dev/null | head -3); allowing stop" >&2
    exit 0
fi
chmod +x "$SCRATCH/compile.sh" 2>/dev/null || true
# Wipe any pre-existing ./executable in the scratch only so we're
# verifying compile.sh actually produces it (rather than a leftover
# from a manual build the agent did inside $WORKSPACE).
rm -f "$SCRATCH/executable"
cd "$SCRATCH"

LOG="$RUNS_DIR/compile.log"
if ! timeout "$TIMEOUT" bash ./compile.sh > "$LOG" 2>&1; then
    rc=$?
    TAIL=$(tail -c 4000 "$LOG" 2>/dev/null || echo "(no output)")
    {
        echo "[compile-contract] ./compile.sh exited non-zero (rc=$rc)."
        echo
        echo "The eval harness will run this exact script from a clean"
        echo "extraction of your tar and reject the submission if it"
        echo "fails. Last 4 KB of compile.sh output:"
        echo
        echo "$TAIL"
        echo
        echo "Fix the build error in compile.sh (or whatever it"
        echo "invokes) and signal completion again."
        echo "(retry $COUNT/$MAX_RETRIES)"
    } >&2
    exit 2
fi

if [ ! -f "./executable" ]; then
    {
        echo "[compile-contract] ./compile.sh exited 0 but ./executable was not produced."
        echo
        echo "The eval harness expects the build script to write the"
        echo "final binary to exactly ./executable in the workspace"
        echo "root (not target/release/foo, not build/foo, etc.). Add"
        echo "a final cp/mv to ./compile.sh, e.g.:"
        echo
        echo "    cp -f target/release/<binname> ./executable"
        echo "    chmod +x ./executable"
        echo
        echo "Then signal completion again."
        echo "(retry $COUNT/$MAX_RETRIES)"
    } >&2
    exit 2
fi

echo "[compile-contract] build contract OK (compile.sh -> ./executable, verified in $SCRATCH)" >&2
exit 0
