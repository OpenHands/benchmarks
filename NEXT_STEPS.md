# Next Steps to Complete GAIA MCP Fix

## ✅ Completed

1. **Root cause identified**: `mcp-server-fetch` download during agent initialization causes 1-18 minute delays
2. **Solution implemented**: Derived Dockerfile that pre-caches MCP server
3. **Code changes committed**: All changes pushed to `openhands/multi-benchmark-eval-support` branch
4. **Documentation created**: Comprehensive README_MCP_FIX.md
5. **Image built locally**: Verified MCP server cached successfully (23ms startup)

## 🔄 Requires Action

### Step 1: Build and Push Image (Requires Registry Permissions)

Someone with `ghcr.io/openhands` registry write access needs to build and push the image:

```bash
# Clone the benchmarks repo and checkout the branch
git clone https://github.com/OpenHands/benchmarks.git
cd benchmarks
git checkout openhands/multi-benchmark-eval-support

# Build the derived image
docker build -f benchmarks/gaia/Dockerfile.gaia \
  --build-arg SDK_IMAGE=ghcr.io/openhands/eval-agent-server:f715937-gaia-binary-minimal \
  -t ghcr.io/openhands/eval-agent-server:f715937-gaia-binary-minimal-with-mcp \
  .

# Login to registry (requires appropriate token)
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Push the image
docker push ghcr.io/openhands/eval-agent-server:f715937-gaia-binary-minimal-with-mcp
```

**Expected time**: ~30 seconds to build, ~2 minutes to push (most layers already exist)

**Verification**:
```bash
# Verify image exists
docker pull ghcr.io/openhands/eval-agent-server:f715937-gaia-binary-minimal-with-mcp

# Verify MCP server is cached
docker run --rm --entrypoint /bin/bash \
  ghcr.io/openhands/eval-agent-server:f715937-gaia-binary-minimal-with-mcp \
  -c "uvx mcp-server-fetch --help 2>&1 | head -3"
```

Should output: `Installed 44 packages in 23ms` (fast startup from cache)

### Step 2: Merge PR (Optional but Recommended)

The GitHub Actions workflow can only be triggered after the workflow file is in the default branch.

1. Create PR from `openhands/multi-benchmark-eval-support` to `main`
2. Get review and merge
3. Future image builds can use the workflow: Actions → "Build GAIA Image with MCP Pre-installed"

### Step 3: Run Test Evaluation

After the image is pushed, run a small test evaluation:

```bash
# Setup GKE access (if not already done)
# ... (see GKE access instructions)

# In the evaluation environment
cd benchmarks
git pull origin openhands/multi-benchmark-eval-support

# Run a small test evaluation (3 instances)
# This will use the new MCP-enabled image automatically
make eval BENCHMARK=gaia SPLIT=validation LIMIT=3
```

### Step 4: Monitor and Verify

Watch the evaluation logs to verify the fix works:

```bash
# Get the job name
kubectl get jobs -n evaluation-jobs --sort-by=.metadata.creationTimestamp | tail -5

# Watch logs for one of the instances
kubectl logs job/<job-name> -n evaluation-jobs -f
```

**What to look for**:
1. Image pull should succeed: `ghcr.io/openhands/eval-agent-server:f715937-gaia-binary-minimal-with-mcp`
2. Conversation creation should be fast (<10 seconds)
3. Look for log line: `"Creating MCP tools"` followed quickly by conversation ready
4. NO timeout errors during conversation creation

**Before Fix** (for comparison):
```
2024-12-04 10:03:39 | INFO | Creating MCP tools...
... 5+ minutes of silence ...
2024-12-04 10:09:12 | INFO | Conversation created
```

**After Fix** (expected):
```
2024-12-04 10:03:39 | INFO | Creating MCP tools...
2024-12-04 10:03:42 | INFO | Conversation created  <-- ~3 seconds later
```

### Step 5: Full Evaluation Run

Once verified on 3 instances, run the full evaluation:

```bash
make eval BENCHMARK=gaia SPLIT=validation
```

Expected improvements:
- No timeout errors during conversation creation
- Consistent startup times (~5-10 seconds)
- Overall evaluation completes faster
- Better resource utilization

## 📊 Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Conversation startup | 1-18 min (variable) | <10 sec (consistent) |
| Timeout rate | ~30-50% | 0% |
| Avg time per instance | Variable | Predictable |
| Image build time | 0 sec | +3 sec |
| Image size | Base | +~50 MB |

## 🚨 Troubleshooting

### "Image does not exist" error

**Symptom**: 
```
RuntimeError: Agent server image ... does not exist in container registry
```

**Solution**: Complete Step 1 - build and push the image

### Still experiencing timeouts

**Check 1**: Verify correct image is being used
```bash
kubectl get pod <pod-name> -o jsonpath='{.spec.containers[0].image}'
```
Should show: `...with-mcp` suffix

**Check 2**: Verify MCP server is cached in the image
```bash
kubectl exec <pod-name> -- bash -c "uvx mcp-server-fetch --help 2>&1 | head -3"
```
Should show: "Installed ... in ~20-30ms"

**Check 3**: Check for other timeout causes
```bash
kubectl logs <pod-name> | grep -i "timeout\|error"
```

### Image build fails

**Check Dockerfile syntax**:
```bash
docker build -f benchmarks/gaia/Dockerfile.gaia \
  --build-arg SDK_IMAGE=ghcr.io/openhands/eval-agent-server:f715937-gaia-binary-minimal \
  --no-cache \
  -t test-mcp .
```

**Verify base image exists**:
```bash
docker pull ghcr.io/openhands/eval-agent-server:f715937-gaia-binary-minimal
```

## 📝 Files Changed

All changes are in branch: `openhands/multi-benchmark-eval-support`

```
.github/workflows/
└── build-gaia-mcp-image.yml          # NEW: Automated build workflow

benchmarks/gaia/
├── Dockerfile.gaia                    # NEW: Derived Dockerfile
├── README_MCP_FIX.md                  # NEW: Detailed documentation
└── run_infer.py                       # MODIFIED: Use -with-mcp suffix

NEXT_STEPS.md                          # NEW: This file
```

## 📚 Additional Resources

- **Detailed documentation**: `benchmarks/gaia/README_MCP_FIX.md`
- **Original timeout analysis**: See previous investigation notes
- **Dockerfile**: `benchmarks/gaia/Dockerfile.gaia` (5 lines)
- **GitHub Actions workflow**: `.github/workflows/build-gaia-mcp-image.yml`

## Contact

If you have questions or need help:
1. Review `benchmarks/gaia/README_MCP_FIX.md`
2. Check the commit message for context
3. Review the Dockerfile - it's only 5 lines!
