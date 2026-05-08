#!/usr/bin/env bash
# Stop hook: refuse to let the agent finish if its binary's output on a
# small set of deterministic probes does not match the reference binary's
# output byte-for-byte.
#
# This script runs inside the agent's container at the agent's `finish`
# action, before /workspace is tarred up as the submission. It compares
#
#     diff <($PB_REFERENCE_BINARY_PATH --help) <(./executable --help)
#     diff <($PB_REFERENCE_BINARY_PATH -h)     <(./executable -h)
#
# (and skips probes the reference doesn't support). Any non-empty diff
# blocks stop and feeds the diff back to the agent as a MessageEvent.
#
# Why this exists, and why it replaces the old `check_gold_tests.sh`:
#
# The previous gold-vs-agent test-comparison hook expected a stashed gold
# binary at `/opt/programbench-stashed-executable-do-not-modify`. That
# path was never populated at cleanroom build time (the upstream task
# images don't ship a stashed gold), so the old hook hit its fail-open
# clause on every run — `gold binary missing; cannot compare; allowing
# stop` — turning the heavy comparison hook into a pass-through.
#
# The reference binary, on the other hand, IS guaranteed to be present:
# the prompt template (``benchmarks/programbench/prompts/default.j2``)
# tells the agent it lives at ``/workspace/<repo_name>`` and the
# cleanroom image ships it there. So we lean on what's actually
# available rather than a path the upstream never provided.
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
#   stdin                              NOT read. The SDK wraps this
#                                      hook via ``bash -s <<EOF ... EOF``
#                                      (see ``run_infer.py::
#                                      _hook_definition_from_script``)
#                                      which makes the heredoc bash's
#                                      stdin -- i.e. the rest of THIS
#                                      script.  Anything that consumes
#                                      stdin here (cat with no args,
#                                      ``exec </dev/null``, ``read``)
#                                      consumes the script source and
#                                      bash silently exits 0 BEFORE
#                                      any check runs, turning the hook
#                                      into a no-op that green-lights
#                                      every broken submission.
#                                      ``test_hooks_do_not_consume_their
#                                      _own_stdin`` pins this.
#
#   PB_REFERENCE_BINARY_PATH           (required-ish) absolute path to
#                                      the reference binary the agent
#                                      is cloning. ``run_infer.py``
#                                      injects this per-instance via
#                                      an env-prelude on the hook
#                                      command. If unset or missing,
#                                      we cannot compare and exit 0.
#   PB_AGENT_BINARY_PATH               (default: ./executable) the
#                                      agent's compiled binary.
#   PB_REFERENCE_DIFFS_RUNS_DIR        (default: /tmp/programbench-ref-diffs)
#                                      MUST be outside $PB_WORKSPACE --
#                                      the orchestrator tars /workspace
#                                      immediately after this hook
#                                      returns, so any churn there
#                                      races with tar and trips
#                                      ``tar: .: file changed as we
#                                      read it``.
#   PB_REFERENCE_DIFFS_MAX_RETRIES     (default: 20) cap re-entries.
#                                      The agent's per-conversation
#                                      iteration budget already bounds
#                                      runaway loops; this only guards
#                                      against a hook whose feedback
#                                      the agent ignores.
#   PB_REFERENCE_DIFFS_TIMEOUT         (default: 30) per-probe timeout
#                                      in seconds.
#   PB_REFERENCE_DIFFS_MAX_DIFF_BYTES  (default: 4000) cap on diff
#                                      bytes piped back to the agent
#                                      so a runaway diff can't bury the
#                                      conversation in a single
#                                      MessageEvent.

set -uo pipefail

WORKSPACE="${PB_WORKSPACE:-/workspace}"
REF="${PB_REFERENCE_BINARY_PATH:-}"
AGENT="${PB_AGENT_BINARY_PATH:-./executable}"
RUNS_DIR="${PB_REFERENCE_DIFFS_RUNS_DIR:-/tmp/programbench-ref-diffs}"
MAX_RETRIES="${PB_REFERENCE_DIFFS_MAX_RETRIES:-20}"
TIMEOUT="${PB_REFERENCE_DIFFS_TIMEOUT:-30}"
MAX_DIFF_BYTES="${PB_REFERENCE_DIFFS_MAX_DIFF_BYTES:-4000}"

mkdir -p "$RUNS_DIR" 2>/dev/null || {
    echo "[ref-diffs] could not create state dir $RUNS_DIR; allowing stop" >&2
    exit 0
}

# --- Retry cap ----------------------------------------------------------
# Same pattern as check_compile_contract.sh: bound re-entries so we never
# burn the entire iteration budget on stop hooks.
RUN_FILE="$RUNS_DIR/run-count"
COUNT=$(cat "$RUN_FILE" 2>/dev/null || echo 0)
COUNT=$((COUNT + 1))
echo "$COUNT" > "$RUN_FILE"
if [ "$COUNT" -gt "$MAX_RETRIES" ]; then
    echo "[ref-diffs] reached max retries ($MAX_RETRIES); allowing stop" >&2
    exit 0
fi

# --- Reference binary availability --------------------------------------
# If the cleanroom image somehow didn't ship the reference binary, we
# can't compare. Fall back to allow-stop and let the upstream eval be
# the source of truth (we've already lost no information vs the old
# fail-open hook).
if [ -z "$REF" ]; then
    echo "[ref-diffs] PB_REFERENCE_BINARY_PATH unset; allowing stop" >&2
    exit 0
fi
if [ ! -f "$REF" ] || [ ! -x "$REF" ]; then
    echo "[ref-diffs] reference binary at $REF is not an executable file; allowing stop" >&2
    exit 0
fi

# --- Agent binary availability ------------------------------------------
# If the agent's binary is missing, the compile-contract hook (which
# always runs first) will block on its own. We don't want to double-block,
# so just allow stop here -- the contract hook owns that error path.
AGENT_RESOLVED="$AGENT"
if [ ! -x "$AGENT_RESOLVED" ]; then
    AGENT_BASENAME="$(basename -- "$AGENT")"
    if [ -x "$WORKSPACE/$AGENT_BASENAME" ]; then
        AGENT_RESOLVED="$WORKSPACE/$AGENT_BASENAME"
    fi
fi
if [ ! -x "$AGENT_RESOLVED" ]; then
    echo "[ref-diffs] agent binary at $AGENT not executable; deferring to compile-contract hook" >&2
    exit 0
fi

# --- Probe definition ---------------------------------------------------
# Probes are flag-args that we expect to be deterministic on both the
# reference and agent binaries. Each probe is run with a hard timeout
# (some agent binaries hang on `--help` if argv parsing is broken, e.g.
# they sit waiting for stdin).
#
# We keep the list short and conservative on purpose: every entry must
# produce stable, byte-comparable output across runs of the SAME binary.
# `--version` is deliberately NOT included by default because most
# binaries embed compile-time timestamps/hashes there, and we'd rather
# under-flag than false-positive.
PROBES=(
    "--help"
    "-h"
)

# Run a probe and capture stdout+stderr. Returns the rc as well.
# We use `timeout --foreground` so SIGTERM propagates to the child if
# the heredoc-wrapped bash gets a signal.
run_probe() {
    local bin="$1"
    local arg="$2"
    # `${arg:+$arg}` lets us pass empty arg (no-op) without a stray ''
    timeout --foreground --kill-after=2 "$TIMEOUT" \
        "$bin" ${arg:+"$arg"} 2>&1
    return $?
}

# --- Run probes ---------------------------------------------------------
# We run probes against the reference FIRST. If the reference itself
# doesn't accept a probe (exits non-zero with no stdout, or hangs to
# timeout), we skip that probe -- there's nothing meaningful to diff
# against. This keeps us conservative: we only block on diffs we're
# confident are real specification mismatches.
DIFFS_FOUND=0
DIFF_BUF=""
SKIPPED=0
COMPARED=0

# A scratch dir for probe outputs. We intentionally land it in /tmp so
# the workspace stays byte-stable through the hook.
SCRATCH=$(mktemp -d "$RUNS_DIR/probe.XXXXXX") || {
    echo "[ref-diffs] could not allocate scratch dir under $RUNS_DIR; allowing stop" >&2
    exit 0
}
trap 'rm -rf "$SCRATCH"' EXIT

for arg in "${PROBES[@]}"; do
    REF_OUT="$SCRATCH/ref$(echo "$arg" | tr -c 'A-Za-z0-9' '_').out"
    AGENT_OUT="$SCRATCH/agent$(echo "$arg" | tr -c 'A-Za-z0-9' '_').out"

    # Probe the reference. Skip on timeout / empty output: nothing to
    # compare against. We deliberately do not require rc=0 -- some
    # binaries return non-zero from --help (e.g. busybox-style usage
    # printed to stderr on rc=1). What we need is *deterministic* output.
    run_probe "$REF" "$arg" > "$REF_OUT" 2>&1
    REF_RC=$?
    REF_BYTES=$(wc -c < "$REF_OUT" 2>/dev/null || echo 0)
    if [ "$REF_RC" = 124 ] || [ "$REF_RC" = 137 ] || [ "$REF_BYTES" -eq 0 ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    run_probe "$AGENT_RESOLVED" "$arg" > "$AGENT_OUT" 2>&1
    AGENT_RC=$?

    if cmp -s "$REF_OUT" "$AGENT_OUT"; then
        COMPARED=$((COMPARED + 1))
        continue
    fi

    COMPARED=$((COMPARED + 1))
    DIFFS_FOUND=$((DIFFS_FOUND + 1))

    # Render a unified diff with the reference as the "expected" side.
    # Bound the size: a runaway diff can drown the agent's context.
    REF_LABEL="reference: $(basename -- "$REF") $arg"
    AGENT_LABEL="agent: $(basename -- "$AGENT_RESOLVED") $arg"
    DIFF=$(diff -u --label "$REF_LABEL" --label "$AGENT_LABEL" \
                "$REF_OUT" "$AGENT_OUT" 2>/dev/null)
    DIFF_LEN=${#DIFF}
    if [ "$DIFF_LEN" -gt "$MAX_DIFF_BYTES" ]; then
        DIFF="${DIFF:0:$MAX_DIFF_BYTES}
... [diff truncated to $MAX_DIFF_BYTES bytes; total was $DIFF_LEN]"
    fi
    DIFF_BUF+="
=== Probe: \`$arg\` (rc: ref=$REF_RC, agent=$AGENT_RC) ===
$DIFF
"
done

# --- Verdict ------------------------------------------------------------
# The error message is rendered via a brace group with a single ``>&2``
# redirect rather than ``cat <<EOF >&2`` because under the SDK's heredoc
# wrap, bash reads THIS script's body from its own stdin -- and the
# top-level-stdin-consumer regex in ``test_hooks_do_not_consume_their_own_
# stdin`` treats any redirection-only ``cat`` invocation as a potential
# heredoc-source-consumer (overcautious, but a useful tripwire that we
# don't want to weaken). Using ``echo`` with positional args sidesteps
# that test trivially.
if [ "$DIFFS_FOUND" -gt 0 ]; then
    {
        echo "[ref-diffs] Your binary's output differs from the reference's on $DIFFS_FOUND of $COMPARED comparable probe(s) (skipped $SKIPPED probes the reference doesn't support)."
        echo ""
        echo "The hidden test suite asserts these character-for-character. Fix EVERY diff"
        echo "below before calling \`finish\` again. Pay special attention to:"
        echo ""
        echo "  * leading/trailing whitespace on every line (a single leading space in"
        echo "    the reference's help banner is a common source of failures)"
        echo "  * banner / preamble lines that you may have added but the reference"
        echo "    does not print (e.g. \"Targeting file ...\", \"Starting N workers ...\")"
        echo "  * line endings, blank lines, and trailing newlines"
        echo "$DIFF_BUF"
        echo ""
        echo "To verify locally:"
        echo ""
        echo "    diff <($REF --help) <($AGENT_RESOLVED --help)"
        echo "    diff <($REF -h)     <($AGENT_RESOLVED -h)"
        echo ""
        echo "Both must print nothing before \`finish\` will be allowed through."
    } >&2
    exit 2
fi

echo "[ref-diffs] all $COMPARED comparable probe(s) match reference (skipped $SKIPPED unsupported); allowing stop" >&2
exit 0
