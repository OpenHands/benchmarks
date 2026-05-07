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
#   exit 1  -> block the stop, print feedback to stderr (the SDK
#              injects it as an environment MessageEvent and resumes
#              the conversation)
#
# Inputs:
#   stdin                          JSON HookEvent (drained, ignored)
#   PB_WORKSPACE                   (default: /workspace) workspace root
#   PB_COMPILE_HOOK_MAX_RETRIES    (default: 3) cap re-entries
#   PB_COMPILE_HOOK_RUNS_DIR       (default:
#                                  $PB_WORKSPACE/.programbench-compile-hook)
#   PB_COMPILE_HOOK_TIMEOUT        (default: 1800) compile.sh timeout secs

set -uo pipefail

WORKSPACE="${PB_WORKSPACE:-/workspace}"
RUNS_DIR="${PB_COMPILE_HOOK_RUNS_DIR:-$WORKSPACE/.programbench-compile-hook}"
MAX_RETRIES="${PB_COMPILE_HOOK_MAX_RETRIES:-3}"
TIMEOUT="${PB_COMPILE_HOOK_TIMEOUT:-1800}"

cd "$WORKSPACE" 2>/dev/null || {
    echo "[compile-contract] cannot cd to $WORKSPACE; allowing stop" >&2
    exit 0
}
mkdir -p "$RUNS_DIR"

# Drain stdin so the SDK doesn't see a SIGPIPE.
cat >/dev/null || true

# --- Retry cap ------------------------------------------------------------
RETRY_FILE="$RUNS_DIR/count"
COUNT=$(cat "$RETRY_FILE" 2>/dev/null || echo 0)
COUNT=$((COUNT + 1))
echo "$COUNT" > "$RETRY_FILE"
if [ "$COUNT" -gt "$MAX_RETRIES" ]; then
    echo "[compile-contract] reached max retries ($MAX_RETRIES); allowing stop" >&2
    exit 0
fi

# --- 1. compile.sh must exist --------------------------------------------
if [ ! -f "./compile.sh" ]; then
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
    exit 1
fi
chmod +x ./compile.sh 2>/dev/null || true

# --- 2. compile.sh must build cleanly and produce ./executable ----------
# Wipe any pre-existing ./executable so we're verifying compile.sh
# actually produces it (rather than a leftover from a manual build).
rm -f ./executable

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
    exit 1
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
    exit 1
fi

if [ ! -x "./executable" ]; then
    chmod +x ./executable 2>/dev/null || true
fi

echo "[compile-contract] build contract OK (compile.sh -> ./executable)" >&2
exit 0
