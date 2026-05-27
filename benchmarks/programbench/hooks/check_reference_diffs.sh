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
#                                      in seconds for the top-level
#                                      --help / -h probes.
#   PB_REFERENCE_DIFFS_BULK_TIMEOUT    (default: 5) per-probe timeout
#                                      for the bulk subcommand probes.
#                                      Help-text and error-text probes
#                                      both complete in <100 ms on any
#                                      sane CLI binary, so 5s is plenty
#                                      and keeps total wall-time bounded
#                                      when there are many subcommands.
#   PB_REFERENCE_DIFFS_MAX_SUBCMDS     (default: 8) max number of
#                                      subcommands to discover and
#                                      probe. Higher values catch more
#                                      drift but lengthen the hook.
#                                      Each subcommand fans out to 3
#                                      probes (--help, invalid flag,
#                                      nonexistent path), so 8 means
#                                      24 subcommand probes in addition
#                                      to the 3 top-level ones.
#   PB_REFERENCE_DIFFS_MAX_DIFF_BYTES  (default: 4000) cap on diff
#                                      bytes piped back to the agent
#                                      so a runaway diff can't bury the
#                                      conversation in a single
#                                      MessageEvent. Per-probe blocks
#                                      are individually capped at 1/3
#                                      of this so several probe blocks
#                                      can coexist before truncation.

set -uo pipefail

WORKSPACE="${PB_WORKSPACE:-/workspace}"
# The cleanroom image ships the reference binary at /workspace/executable
# with mode ---x--x--x (execute-only). The agent's compile.sh must produce
# a binary at /workspace/executable, so the agent is told (in the very
# first directive of prompts/default.j2) to ``mv /workspace/executable
# /workspace/executable.ref`` before doing anything else. The default
# below matches that convention; tests and ad-hoc callers can override
# via PB_REFERENCE_BINARY_PATH.
REF="${PB_REFERENCE_BINARY_PATH:-/workspace/executable.ref}"
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
# If the agent never ran the prompted ``mv /workspace/executable
# /workspace/executable.ref``, the reference path won't exist (the
# agent's compile.sh has by now overwritten /workspace/executable
# with their own build, so we can't fall back to that). Fall back to
# allow-stop and let the upstream eval be the source of truth — we've
# lost no information vs the old fail-open gold-tests hook.
if [ ! -f "$REF" ] || [ ! -x "$REF" ]; then
    echo "[ref-diffs] reference binary at $REF is not an executable file" >&2
    echo "[ref-diffs] (the prompt's Step 0 'mv /workspace/executable" >&2
    echo "[ref-diffs] /workspace/executable.ref' likely did not run); allowing stop" >&2
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

# --- Probe configuration ------------------------------------------------
# v2 (post-retry-22): we run THREE families of probes:
#
#   1. Top-level help banners        (--help, -h)
#         catches: leading-whitespace drift, banner re-flow, version-string
#         drift, hardcoded-program-name vs argv[0] drift
#   2. Top-level invalid input       (--<bogus-flag>)
#         catches: agent silently accepts unknown flags / wrong rc on errors
#   3. Per-subcommand probes          (<sub> --help, <sub> --<bogus>,
#                                      <sub> /<bogus-path>)
#         catches: missing subcommands, drifted subcommand help banners,
#         missing input validation (most R22 zoxide failures were here)
#
# Subcommands are discovered by parsing the reference's top-level --help
# output. We use a heuristic regex that matches clap, argparse, cobra,
# and similar frameworks' "Commands:"/"Subcommands:"/"Available Commands:"
# section formats. We cap to PB_REFERENCE_DIFFS_MAX_SUBCMDS (default 8)
# subcommands to bound the hook's runtime budget.
#
# Per-probe timeout for the BULK probes (subcommand stuff) is shorter
# than the original PB_REFERENCE_DIFFS_TIMEOUT to keep total wall-time
# bounded. Help-text and error-text probes both complete in <100 ms on
# any sane CLI binary, so 5 s is plenty.
MAX_SUBCMDS="${PB_REFERENCE_DIFFS_MAX_SUBCMDS:-8}"
BULK_TIMEOUT="${PB_REFERENCE_DIFFS_BULK_TIMEOUT:-5}"
# A long, clearly-synthetic flag name and path that no real CLI should
# accept. Identical strings used for ref and agent to keep the probe
# input symmetric.
BOGUS_FLAG="--__openhands_invalid_test_flag__"
BOGUS_PATH="/tmp/__openhands_test_nonexistent_path_do_not_create__"
# Common argv[0] for both binaries. Many CLI frameworks (clap default,
# argparse default) embed argv[0] in their "Usage: <name> ..." banner,
# so if we invoke the reference as ``executable.ref`` and the agent as
# ``executable`` the diff is full of basename noise even when both
# binaries are correct. By exec'ing each with the SAME argv[0] we make
# the comparison genuinely about behaviour: matching outputs mean
# matching behaviour, divergent outputs mean a real bug (e.g. agent
# hardcoded ``Command::name("zoxide")`` while reference uses argv[0]).
# We pick "executable" because that's the name the hidden test harness
# also uses, so the agent's binary is being probed under the exact
# argv[0] the test suite will pass it.
ARGV0_NAME="${PB_REFERENCE_DIFFS_ARGV0_NAME:-executable}"

# --- Probe runner -------------------------------------------------------
# Run a probe (multi-token args) and capture stdout+stderr. Returns rc.
# We use `timeout --foreground` so SIGTERM propagates to the child if
# the heredoc-wrapped bash gets a signal.
#
# We invoke the binary via ``bash -c 'exec -a "$1" "${@:2}"' _ "$ARGV0"
# "$bin" "$@"`` so argv[0] for the child is ARGV0_NAME rather than
# $bin's actual path. This removes filename noise from the diff (see
# ARGV0_NAME comment above).
#
# CRITICAL: stdin is redirected from ``/dev/null``. The SDK invokes
# this script via ``bash -s <<'EOF' ... EOF`` (see ``run_infer.py::
# _hook_definition_from_script``) which makes the script body itself
# the parent bash's stdin. Any descendant that READS stdin will eat
# script body bytes from there, corrupting the rest of the hook. We
# already audit-pin "this script doesn't read stdin" in
# ``test_hooks_do_not_consume_their_own_stdin``, but with v2 we now
# probe arbitrary subcommands of the agent's binary -- one of which
# could plausibly read stdin (think ``cat``/``grep`` semantics where
# bare invocation means "read stdin"). ``< /dev/null`` insulates the
# script body from any such probe. Without this redirect, retry-23
# hung for 6 hours waiting for inference output -- a probe consumed
# part of the script body and the resulting hook returned garbage,
# but exit 1 is silently ignored by the SDK and the agent kept
# iterating until the workflow's wall-clock timeout.
#
# Args: $1 = bin path, $2 = per-probe timeout (seconds), $3.. = argv to bin
run_probe() {
    local bin="$1"
    local probe_timeout="$2"
    shift 2
    timeout --foreground --kill-after=2 "$probe_timeout" \
        bash -c 'exec -a "$1" "${@:2}"' \
        _ "$ARGV0_NAME" "$bin" "$@" </dev/null 2>&1
    return $?
}

# --- Subcommand discovery -----------------------------------------------
# Parse top-level --help output for a "Commands:" / "Subcommands:" /
# "Available Commands:" / "Available subcommands:" section and emit the
# names of subcommands one per line (deduped, capped). The heuristic is
# deliberately strict to avoid false positives:
#   * Section header line matches one of the known headers (case-insens)
#   * Subcommand lines are indented and start with a name token of
#     [a-z][a-z0-9_-]* up to 30 chars, followed by whitespace
#   * The 'help' subcommand (clap auto-adds it) is filtered out
#   * Section ends on first non-indented line (typically the next
#     section header)
discover_subcommands() {
    awk -v max="$MAX_SUBCMDS" '
        BEGIN { in_section = 0; emitted = 0 }
        # Section start: only the canonical headers, case-insensitive.
        # Match the ENTIRE line so a stray "Commands:" inside a
        # description (e.g. "Commands: foo, bar") does not trip us.
        tolower($0) ~ /^(commands|subcommands|available commands|available subcommands):[[:space:]]*$/ {
            in_section = 1; next
        }
        # Section end: any non-indented line (header for the next
        # section, or the trailing "Use --help on a subcommand for
        # more information." footer).
        in_section && /^[^[:space:]]/ {
            in_section = 0
            next
        }
        # Subcommand line: indented, then a [a-z]-leading short token,
        # then whitespace (description). The strict character class
        # avoids matching e.g. "* --foo" bullet entries that some
        # frameworks emit.
        in_section && match($0, /^[[:space:]]+[a-z][a-z0-9_-]{0,29}[[:space:]]/) {
            line = substr($0, RSTART, RLENGTH)
            sub(/^[[:space:]]+/, "", line)
            sub(/[[:space:]].*$/, "", line)
            if (line == "help") next         # clap auto-help
            if (line in seen)    next
            seen[line] = 1
            print line
            emitted++
            if (emitted >= max) exit
        }
    '
}

# --- Probe loop ---------------------------------------------------------
# We run probes against the reference FIRST. If the reference itself
# doesn't accept a probe (exits non-zero with no stdout, or hangs to
# timeout), we skip that probe -- there's nothing meaningful to diff
# against. This keeps us conservative: we only block on diffs we're
# confident are real specification mismatches.
DIFFS_FOUND=0
DIFF_BUF=""
SKIPPED=0
COMPARED=0
TOTAL_DIFF_BYTES=0

# A scratch dir for probe outputs. We intentionally land it in /tmp so
# the workspace stays byte-stable through the hook.
SCRATCH=$(mktemp -d "$RUNS_DIR/probe.XXXXXX") || {
    echo "[ref-diffs] could not allocate scratch dir under $RUNS_DIR; allowing stop" >&2
    exit 0
}
trap 'rm -rf "$SCRATCH"' EXIT

# Compare a single probe and accumulate diffs into DIFF_BUF.
# Args: $1 = label (for diff buffer), $2 = per-probe timeout,
#       $3.. = argv tokens to pass to BOTH binaries.
compare_probe() {
    local label="$1"
    local probe_timeout="$2"
    shift 2

    # Filename-safe slug from the label
    local slug
    slug=$(echo "$label" | tr -c 'A-Za-z0-9' '_' | head -c 40)
    local ref_out="$SCRATCH/ref_${slug}.out"
    local agent_out="$SCRATCH/agent_${slug}.out"

    # Reference first. Skip silently on timeout/empty -- not a meaningful
    # comparison target.
    run_probe "$REF" "$probe_timeout" "$@" > "$ref_out" 2>&1
    local ref_rc=$?
    local ref_bytes
    ref_bytes=$(wc -c < "$ref_out" 2>/dev/null || echo 0)
    if [ "$ref_rc" = 124 ] || [ "$ref_rc" = 137 ] || [ "$ref_bytes" -eq 0 ]; then
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    run_probe "$AGENT_RESOLVED" "$probe_timeout" "$@" > "$agent_out" 2>&1
    local agent_rc=$?
    COMPARED=$((COMPARED + 1))

    # rc must match AND bytes must match. We split these so the diff
    # message can clearly call out rc-only mismatches (a common pattern
    # for "agent forgot to validate; returns 0 where ref returns 1").
    local rc_match=1
    local bytes_match=1
    [ "$ref_rc" != "$agent_rc" ] && rc_match=0
    cmp -s "$ref_out" "$agent_out" || bytes_match=0
    if [ "$rc_match" = 1 ] && [ "$bytes_match" = 1 ]; then
        return 0
    fi
    DIFFS_FOUND=$((DIFFS_FOUND + 1))

    # If we've already buffered enough diff content, drop further
    # probe-detail blocks and just count -- the agent has plenty to act
    # on already. We still do the comparison so the count is honest.
    if [ "$TOTAL_DIFF_BYTES" -gt "$MAX_DIFF_BYTES" ]; then
        return 0
    fi

    # Render a unified diff with the reference as the "expected" side.
    # Both binaries were invoked with argv[0]="$ARGV0_NAME" via exec -a,
    # so we use that name in the labels rather than the actual path on
    # disk -- it makes the diff match what the agent will see when they
    # re-run the same probe locally against ``./executable``.
    local ref_label="reference: $ARGV0_NAME"
    local agent_label="agent: $ARGV0_NAME"
    [ $# -gt 0 ] && {
        ref_label="$ref_label $*"
        agent_label="$agent_label $*"
    }
    local diff_text
    diff_text=$(diff -u --label "$ref_label" --label "$agent_label" \
                     "$ref_out" "$agent_out" 2>/dev/null)
    local diff_len=${#diff_text}
    # Per-probe cap is 1/3 of the global cap, leaving room for several
    # probe blocks to coexist before truncation kicks in.
    local per_probe_cap=$((MAX_DIFF_BYTES / 3))
    if [ "$diff_len" -gt "$per_probe_cap" ]; then
        diff_text="${diff_text:0:$per_probe_cap}
... [diff truncated to $per_probe_cap bytes; total was $diff_len]"
        diff_len="$per_probe_cap"
    fi
    TOTAL_DIFF_BYTES=$((TOTAL_DIFF_BYTES + diff_len))
    DIFF_BUF+="
=== $label (rc: ref=$ref_rc, agent=$agent_rc) ===
$diff_text
"
}

# --- 1. Top-level help banners ------------------------------------------
compare_probe "Probe: \`--help\`" "$TIMEOUT" --help
compare_probe "Probe: \`-h\`"     "$TIMEOUT" -h

# --- 2. Top-level invalid-input probe -----------------------------------
# Catches "agent silently accepts unknown flag" and "agent returns rc=0
# where reference rc=1+". The bogus flag is long enough that no real
# CLI should accept it.
compare_probe "Probe: top-level invalid flag (\`$BOGUS_FLAG\`)" \
    "$BULK_TIMEOUT" "$BOGUS_FLAG"

# --- 3. Subcommand probes ------------------------------------------------
# Capture the reference's top-level --help once and parse subcommand names
# out of its "Commands:" section. We deliberately re-run the reference
# here (rather than reusing the file from probe #1) because that file
# already had the diff suffix munging applied; we need the raw stdout.
TOPLEVEL_HELP_OUT="$SCRATCH/_toplevel_help_for_discovery.out"
run_probe "$REF" "$TIMEOUT" --help > "$TOPLEVEL_HELP_OUT" 2>/dev/null || true
SUBCMDS=$(discover_subcommands < "$TOPLEVEL_HELP_OUT")

if [ -n "$SUBCMDS" ]; then
    # We iterate one subcommand per line. ${SUBCMDS} is multi-line.
    while IFS= read -r sub; do
        [ -z "$sub" ] && continue
        compare_probe "Probe: subcommand \`$sub --help\`" \
            "$BULK_TIMEOUT" "$sub" --help
        compare_probe "Probe: subcommand \`$sub\` invalid flag" \
            "$BULK_TIMEOUT" "$sub" "$BOGUS_FLAG"
        compare_probe "Probe: subcommand \`$sub\` nonexistent path" \
            "$BULK_TIMEOUT" "$sub" "$BOGUS_PATH"
    done <<< "$SUBCMDS"
fi

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
        echo "below before calling \`finish\` again. Common drift patterns:"
        echo ""
        echo "  * leading/trailing whitespace on every line (a single leading space in"
        echo "    the reference's help banner is a common source of failures)"
        echo "  * hardcoded program name in clap/argparse vs argv[0] (clap's"
        echo "    \`Command::name(\"foo\")\` overrides argv[0]; the original probably"
        echo "    used the default; \`Command::new(env!(\"CARGO_BIN_NAME\"))\` is fine"
        echo "    BUT only if your binary is built with the matching name)"
        echo "  * banner / preamble lines that you may have added but the reference"
        echo "    does not print (e.g. \"Targeting file ...\", \"Starting N workers ...\")"
        echo "  * line endings, blank lines, and trailing newlines"
        echo "  * SUBCOMMAND validation: many subcommands must rc=1 (NOT rc=0) on"
        echo "    invalid input — e.g., a directory subcommand passed a regular file,"
        echo "    a remove subcommand passed a path that isn't tracked, an import"
        echo "    subcommand passed a malformed file. The reference rejects; if your"
        echo "    binary silently succeeds you'll see ref=1 vs agent=0 below"
        echo "  * INVALID-FLAG handling: every CLI framework rejects unknown flags."
        echo "    If the agent's probe shows ref=2 vs agent=0 on the bogus flag,"
        echo "    you have an argv parser leak"
        echo "  * CASE-INSENSITIVE enum values: many flags accept any case for their"
        echo "    enum values (e.g. \`-C RED\` == \`-C red\`). Try uppercase, mixed"
        echo "    case, and lowercase against the reference; mirror that in your"
        echo "    parser"
        echo "$DIFF_BUF"
        echo ""
        echo "To verify locally, re-run the same probes against your build:"
        echo ""
        echo "    diff <($REF --help) <($AGENT_RESOLVED --help)"
        echo "    diff <($REF -h)     <($AGENT_RESOLVED -h)"
        echo "    # plus each subcommand listed in your reference's \"Commands:\" section:"
        echo "    diff <($REF <sub> --help) <($AGENT_RESOLVED <sub> --help)"
        echo "    diff <($REF $BOGUS_FLAG 2>&1; echo rc=\$?) \\"
        echo "         <($AGENT_RESOLVED $BOGUS_FLAG 2>&1; echo rc=\$?)"
        echo ""
        echo "All must print nothing before \`finish\` will be allowed through."
    } >&2
    exit 2
fi

echo "[ref-diffs] all $COMPARED comparable probe(s) match reference (skipped $SKIPPED unsupported); allowing stop" >&2
exit 0
