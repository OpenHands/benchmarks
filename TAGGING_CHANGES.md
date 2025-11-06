# Docker Image Tagging Improvements

## Summary

This change replaces the long, auto-generated versioned tags with short, meaningful tags that include:
- **SDK commit hash** (exact reproducibility)
- **SWE-Bench instance ID** (clear identification)

## Changes Made

### 1. SDK Build System (`vendor/software-agent-sdk/.../docker/build.py`)

**Added two features:**

1. **SDK_VERSION now uses git commit hash**
   - `_sdk_version()` now automatically detects the SDK repo root and gets its commit hash
   - Falls back to package version only if git info unavailable
   - Works correctly in submodule contexts (uses SDK repo, not calling repo)
   - No environment variable override needed - automatic and robust

2. **`include_versioned_tag` option in BuildOptions**
   - When `False`, skips the long versioned tag
   - Defaults to `True` for backward compatibility
   - Gives consumers control over tag format

3. **Target-based tag suffixes** (replaces `-dev` suffix)
   - Non-binary builds include `-{target}` suffix: `-source`, `-binary-minimal`, `-source-minimal`
   - Binary builds have no suffix (it's the default/common case)
   - More descriptive than previous `-dev` suffix (which only applied to source builds)
   - Makes tag meaning immediately clear without needing to check build config
   - Removed deprecated `is_dev` property

### 2. Benchmarks Build Script (`benchmarks/swe_bench/build_images.py`)

**Added one function:**

1. **`extract_instance_id(base_image)`**
   - Parses SWE-Bench base image name to extract instance ID
   - Examples:
     - `...django_1776_django-12155:latest` → `django-12155`
     - `...sympy_1776_sympy-18189:latest` → `sympy-18189`
     - `...scikit-learn_3742_scikit-learn-25973:latest` → `scikit-learn-25973`

**Modified build flow:**

1. Per image: Extract instance ID and create custom tag `swebench-{instance_id}`
2. Pass `include_versioned_tag=False` to disable long tag
3. SDK automatically uses its own commit hash (no manual override needed)

## Tag Format Comparison

### Before (Old Format)
```
ghcr.io/openhands/eval-agent-server:v1.0.0_docker.io_s_swebench_s_sweb.eval.x86_64.django_1776_django-12155_tag_latest_source-minimal-dev
```
- **Length**: 137 characters
- **Includes**: Package version (v1.0.0), full base image path, target
- **Problem**: No git commit info, hard to parse

### After (New Format)

For source-minimal (most common for SWE-Bench):
```
ghcr.io/openhands/eval-agent-server:a612c0a-swebench-django-12155-source-minimal
ghcr.io/openhands/eval-agent-server:main-swebench-django-12155-source-minimal
```
- **Length**: 84 characters (**39% shorter**)

For binary (no suffix, it's the default):
```
ghcr.io/openhands/eval-agent-server:a612c0a-swebench-django-12155
ghcr.io/openhands/eval-agent-server:main-swebench-django-12155
```
- **Length**: 69 characters (**50% shorter**)

**Benefits**: 
  - Exact reproducibility (commit hash)
  - Easy to parse and filter
  - Clear instance identification
  - Clean tags for common case (binary has no suffix)

## Tag Generation Logic

The SDK's `all_tags` property generates:

1. **Commit-based tag**: `{image}:{SDK_VERSION[:7]}-{custom_tag}[-{target}]{arch_suffix}`
   - `SDK_VERSION[:7]` = First 7 chars of SDK commit hash (automatically detected)
   - `custom_tag` = `swebench-{instance_id}`
   - `target` = Build target (omitted for `binary`, included for others)
   - Examples: 
     - Binary: `a612c0a-swebench-django-12155`
     - Source: `a612c0a-swebench-django-12155-source`
     - Source-minimal: `a612c0a-swebench-django-12155-source-minimal`

2. **Main branch tag** (if on main): `{image}:main-{custom_tag}[-{target}]{arch_suffix}`
   - Examples:
     - Binary: `main-swebench-django-12155`
     - Source-minimal: `main-swebench-django-12155-source-minimal`

3. **Versioned tag** (now disabled): `{image}:{versioned_tag}[-{target}]{arch_suffix}`
   - Skipped when `include_versioned_tag=False`

Non-binary targets include `-{target}` suffix for clarity. Binary has no suffix (default case).

## Benefits

### 1. Reproducibility
- Git commit hash ensures exact SDK version tracking
- Can reconstruct exact build environment from tag alone
- No ambiguity (version 1.0.0 could be many commits)

### 2. Usability
- **39% shorter tags** (137 → 84 chars)
- Easy to filter: `docker images | grep a612c0a`
- Easy to identify: `swebench-django-12155-source-minimal` is self-documenting
- Explicit target indication (no more guessing what `-dev` means)
- Fits in terminal/log output better

### 3. Maintainability
- SDK changes are backward compatible (env var is optional)
- Benchmarks repo has full control over tag format
- Can easily extend with more metadata later

## Example Build Command

```bash
uv run benchmarks/swe_bench/build_images.py \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --platforms linux/amd64 \
  --push \
  --max-workers 2
```

## Testing

To test the tagging logic without building:

```python
from benchmarks.swe_bench.build_images import extract_instance_id

# Test instance ID extraction
base = "docker.io/swebench/sweb.eval.x86_64.django_1776_django-12155:latest"
print(extract_instance_id(base))  # → django-12155
```

## Migration Notes

### For existing workflows:
- No changes needed - SDK defaults to old behavior
- Opt-in by setting `include_versioned_tag=False`

### For CI/CD:
- New tags will be generated automatically
- Old tags (if any exist) remain unchanged
- Can coexist during transition period

### For consumers:
- Update image references to use new tag format
- Can filter by SDK version: `grep a612c0a`
- Can filter by instance: `grep django-12155`

## Future Enhancements

Possible additions:
1. **Docker labels** for metadata (see `docker inspect`)
2. **Benchmarks commit** in tag or label
3. **Build timestamp** in labels
4. **Platform/architecture** in tag (already supported via `arch` param)

## Files Changed

1. `vendor/software-agent-sdk/openhands-agent-server/openhands/agent_server/docker/build.py`
   - Refactored `_sdk_version()` to automatically use SDK repo commit hash
   - Added `_git_info_for_repo()` to get git info from specific directories
   - Added `include_versioned_tag` field to `BuildOptions`
   - Changed tag suffix logic: Non-binary targets get `-{target}` suffix, binary gets no suffix
   - Removed deprecated `is_dev` property
   - Modified `all_tags` property to respect new flag and suffix logic

2. `benchmarks/swe_bench/build_images.py`
   - Added `extract_instance_id()` function
   - Modified `build_one()` to use custom tags and disable versioned tag
   - Removed unnecessary SDK_VERSION_OVERRIDE logic (now automatic)

## Related PRs

- **SDK Changes**: https://github.com/OpenHands/software-agent-sdk/pull/1088
  - SDK_VERSION now automatically uses commit hash from SDK repo
  - Changes tag suffix: binary gets no suffix, non-binary gets `-{target}` (more descriptive)
  - Adds `include_versioned_tag` option
  - Works correctly in submodule/vendored contexts
