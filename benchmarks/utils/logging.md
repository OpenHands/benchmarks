# Rich Logging

## Enabling Rich Logs

Set the environment variable:

```bash
export RICH_LOGGING=1
```

When disabled (default), the original logging style is used.

## Log Types

### 1. Startup Log

Displayed when an instance starts processing:

```
2025-02-03 10:30:45 [django-12345]  START  instance_id | Logs: /path/to/logs/instance_xxx.log
```

### 2. Trajectory Logs (Tool Calls)

Shows agent actions in real-time:

```
10:30:45 [django-12345]  TOOL   │ ▶ bash #1 cmd='ls -la'
10:30:46 [django-12345]  TOOL   │   └─ ok
10:30:47 [django-12345]  TOOL   │ ▶ str_replace_editor #2 path='/workspace/file.py'
10:30:48 [django-12345]  WARN   │   └─ exit=1
```

- `#N` = Tool call counter (1st call, 2nd call, ...)
- `└─ ok` = Tool succeeded (exit_code=0)
- `└─ exit=N` = Tool failed with exit code N
- `└─ tool_error` = Tool returned an error

### 3. Message Logs

Shows agent text messages:

```
10:30:50 [django-12345]  MESSAGE│ I see the issue now! Looking at the test data:  1. band_duo ...
```

### 4. Error Logs

Shows agent-side errors:

```
10:30:49 [django-12345]  ERROR  │   └─ error
```

### 5. Summary Log

Displayed at the end of each instance evaluation

**For benchmarks with git patches** (e.g. SWE-bench):

```
OK patch=NONEMPTY commit=0 changes=Y msgs(a/u)=8/3 tool_calls=12 errors(agent/conv)=0/0 end=finish_tool preview='diff --git ...'
```

**For benchmarks without git patches** (GAIA, OpenAgentSafety):

```
OK msgs(a/u)=8/3 tool_calls=12 errors(agent/conv)=0/0 end=finish_tool
```

#### Summary Fields

| Field | Description |
|-------|-------------|
| `OK` / `WITH_ISSUES` | Health status (green/yellow). WITH_ISSUES if errors occurred or status is ERROR |
| `patch=NONEMPTY/EMPTY` | Whether the agent produced a non-empty git patch |
| `commit=N` | Git commit exit code (0 = success) |
| `changes=Y/N` | Whether repo had uncommitted changes after agent run |
| `msgs(a/u)=N/M` | Count of agent/user messages |
| `tool_calls=N` | Total number of tool calls made |
| `errors(agent/conv)=N/M` | Count of agent errors / conversation errors |
| `end=...` | How the run ended: `finish_tool`, `status=ERROR`, `finished_no_finish_tool` |
| `preview='...'` | First ~180 chars of the git diff (grey, truncated) |

## Color Coding

- **Green**: Success / healthy values
- **Yellow**: Warnings (non-zero exit codes, errors present, no finish tool)
- **Red**: Errors
- **Grey/Dim**: Metadata, previews, timestamps

## Disabling Colors

Set `NO_COLOR=1` to disable ANSI color codes in output.

## File Logging

Rich logging only affects console output. File logging behavior is unchanged:
- Full logs are written to `logs/instance_<id>.log`
- Stdout/stderr captured in `logs/instance_<id>.output.log`

